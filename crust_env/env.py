"""
CRust Migration Environment — env.py

Full Markov Decision Process (MDP) implementation for the OpenEnv hackathon.
Trains an LLM agent to migrate legacy C codebases to memory-safe, modular Rust
through dependency-aware topological scheduling and multi-objective verifiable rewards.

Aligned with Hackathon Theme #2: Super Long-Horizon Planning & Instruction Following.
"""

from typing import Dict, Any, List, Optional, Tuple
import uuid
import os
import re

from .verifier import CRustVerifier
from .orchestrator import SemanticOrchestrator
from .metrics import ModularityMetrics
import json
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
except ImportError:
    pass

try:
    from openenv.core.env_server.interfaces import Environment as _OpenEnvBase
except ImportError:
    class _OpenEnvBase:  # type: ignore
        def reset(self, **kwargs): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        @property
        def state(self): raise NotImplementedError


class MigrationEnv(_OpenEnvBase):
    DEFAULT_CONSTRAINTS: List[str] = [
        "Do not use the unsafe keyword",
        "Maintain a CBO score below 3",
    ]

    W_COMPILATION   = 0.30
    W_TESTS         = 0.30
    W_MEMORY_SAFE   = 0.20
    W_CBO           = 0.10
    W_COHESION      = 0.10

    P_UNSAFE_USED   = 0.50
    P_HIGH_CBO      = 0.20

    PROCESS_REWARD_PER_ERROR_CLEARED = 0.02

    def __init__(self, workspace_dir: str, legacy_dir: Optional[str] = None):
        self.workspace_dir = workspace_dir
        self.legacy_dir = legacy_dir or os.path.normpath(
            os.path.join(workspace_dir, "..", "legacy_c")
        )
        self.session_id = str(uuid.uuid4())
        self.verifier = CRustVerifier(workspace_dir)

        self._current_state: Dict[str, Any] = {"status": "uninitialized"}
        self._constraints: List[str] = []
        self._step_count: int = 0
        self._max_steps: int = 200
        self._phase: int = 1

        self._schedule: List[Dict[str, str]] = []
        self._current_idx: int = 0
        self._validation_phase: str = "transpile" # "transpile" or "refactor"
        self._translated: Dict[str, str] = {}
        self._error_history: List[Dict] = []
        self._prev_error_count: int = 0
        
        # FAISS Epistemic Context Engine
        self.embedding_model = None
        self.faiss_index = None
        self.faiss_mapping: List[str] = []
        try:
            self.embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            self.faiss_index = faiss.IndexFlatL2(384) # BGE-small output dim
        except Exception as e:
            print(f"[env.py] FAISS/SentenceTransformer not available: {e}")

    def reset(
        self,
        constraints: Optional[List[str]] = None,
        phase: int = 1,
    ) -> Dict[str, Any]:
        self._step_count = 0
        self._phase = max(1, min(4, phase))
        self._constraints = list(constraints) if constraints else list(self.DEFAULT_CONSTRAINTS)
        self._translated = {}
        self._error_history = []
        self._prev_error_count = 0
        self._validation_phase = "transpile"

        # Epistemic Context Engine reset
        if self.faiss_index:
            self.faiss_index.reset()
        self.faiss_mapping = []

        orchestrator = SemanticOrchestrator(self.legacy_dir, self.workspace_dir)
        manifest_path = orchestrator.generate_scaffolding()
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
            
        full_schedule = [{"name": m["node_name"], "file": m["source_c_file"]} for m in manifest["modules"]]

        if self._phase == 1:
            self._schedule = full_schedule[:1]
        elif self._phase == 2:
            self._schedule = full_schedule[:2]
        elif self._phase == 3:
            self._schedule = full_schedule[:3]
        else:
            self._schedule = full_schedule

        self._current_idx = 0

        self._current_state = {
            "status": "ready",
            "phase": self._phase,
            "validation_phase": self._validation_phase,
            "schedule": self._schedule,
            "current_idx": self._current_idx,
            "files_total": len(self._schedule),
            "files_done": 0,
            "translated_files": [],
            "metrics": {"cbo": 0, "lcom": 0},
            "constraints": self._constraints,
            "error_history": [],
            "step_count": 0,
        }
        return self.observation()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step_count += 1
        self._current_state["step_count"] = self._step_count

        if self._step_count >= self._max_steps:
            return self._format_response(
                done=True, reward=0.01,
                info={"reason": "max_steps_exceeded", "step": self._step_count}
            )

        file_path = (action.get("file_path") or "").strip()
        code_content = (action.get("code_content") or "").strip()

        if not file_path or not code_content:
            return self._format_response(
                done=False, reward=0.01,
                info={"error": "Invalid action: file_path and code_content are required."}
            )

        verification = self.verifier.verify(action)
        metrics = ModularityMetrics.evaluate(code_content)
        self._current_state["metrics"] = metrics

        reward, breakdown = self._compute_reward(code_content, verification, metrics)

        current_errors = [d for d in verification.get("diagnostics", []) if d.get("level") == "error"]
        errors_cleared = max(0, self._prev_error_count - len(current_errors))
        process_reward = errors_cleared * self.PROCESS_REWARD_PER_ERROR_CLEARED
        reward = min(0.99, reward + process_reward)
        self._prev_error_count = len(current_errors)
        self._error_history = verification.get("diagnostics", [])[-10:]

        success = verification.get("success", False)
        episode_done = False

        if success:
            if self._validation_phase == "transpile":
                # In transpile phase, success means it compiles and passes tests (even if unsafe is used)
                self._validation_phase = "refactor"
                self._current_state["validation_phase"] = self._validation_phase
            elif self._validation_phase == "refactor":
                # Check if unsafe is actually removed (handled by compute_reward logic, but here we advance)
                unsafe_count = verification.get("unsafe_count", 0)
                if unsafe_count == 0 or not any("unsafe" in c.lower() for c in self._constraints):
                    self._translated[file_path] = code_content
                    
                    # Epistemic Context Engine: Embed successful translation
                    if self.embedding_model and self.faiss_index:
                        embedding = self.embedding_model.encode([code_content])[0]
                        self.faiss_index.add(np.array([embedding]))
                        self.faiss_mapping.append(code_content)
                        
                    self._current_idx += 1
                    self._validation_phase = "transpile"
                    self._current_state["files_done"] = self._current_idx
                    self._current_state["translated_files"] = list(self._translated.keys())
                    self._current_state["validation_phase"] = self._validation_phase

                    if self._current_idx >= len(self._schedule):
                        episode_done = True

        self._current_state.update({
            "current_idx": self._current_idx,
            "error_history": self._error_history,
        })

        info = {
            "step": self._step_count,
            "verification": verification,
            "metrics": metrics,
            "reward_breakdown": breakdown,
            "process_reward": round(process_reward, 4),
            "errors_cleared": errors_cleared,
            "files_done": self._current_idx,
            "files_total": len(self._schedule),
        }

        return self._format_response(episode_done, reward, info)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            **self._current_state,
            "session_id": self.session_id,
            "workspace_dir": self.workspace_dir,
        }

    def observation(self) -> Dict[str, Any]:
        target = self._get_current_target()
        c_source = target.get("code", "") if target else ""
        target_name = target.get("name", "") if target else ""
        dep_context = self._get_dependency_context()

        # Define the HRL Role
        hrl_role = "Manager: Analyze dependencies and plan" if self._validation_phase == "transpile" else "Worker: Remove unsafe blocks and refactor"

        return {
            "current_target": target_name,
            "c_source_code": c_source,
            "constraints": self._constraints,
            "recent_errors": self._error_history[-5:],
            "dependency_context": dep_context,
            "phase": self._phase,
            "validation_phase": self._validation_phase,
            "hrl_role": hrl_role,
            "files_remaining": max(0, len(self._schedule) - self._current_idx),
            "step": self._step_count,
        }

    def _get_current_target(self) -> Optional[Dict[str, str]]:
        if self._current_idx < len(self._schedule):
            return self._schedule[self._current_idx]
        return None

    def _get_dependency_context(self) -> Dict[str, str]:
        context: Dict[str, str] = {}
        target = self._get_current_target()
        target_name = target.get("name", "") if target else ""
        
        # If FAISS is available and we have translated units, perform semantic retrieval
        if self.embedding_model and self.faiss_index and self.faiss_index.ntotal > 0:
            query_embedding = self.embedding_model.encode([target_name])[0]
            k = min(3, self.faiss_index.ntotal) # Top-3 most relevant translations
            distances, indices = self.faiss_index.search(np.array([query_embedding]), k)
            
            for idx in indices[0]:
                if idx >= 0 and idx < len(self.faiss_mapping):
                    code = self.faiss_mapping[idx]
                    sigs = re.findall(r'pub fn\s+\w+[^{]+', code)
                    pub_types = re.findall(r'pub (?:struct|enum|type)\s+\w+[^{;]*', code)
                    context[f"retrieved_context_{idx}"] = "\\n".join(sigs + pub_types)
        else:
            # Fallback to appending all translated logic
            for fname, code in self._translated.items():
                sigs = re.findall(r'pub fn\s+\w+[^{]+', code)
                pub_types = re.findall(r'pub (?:struct|enum|type)\s+\w+[^{;]*', code)
                context[fname] = "\\n".join(sigs + pub_types)
                
        return context

    def _compute_reward(
        self, code: str, verification: Dict, metrics: Dict
    ) -> Tuple[float, Dict]:
        reward = 0.0
        breakdown: Dict[str, float] = {}

        stage = verification.get("stage", "")
        success = verification.get("success", False)
        compiled = stage in ("testing", "complete")
        diag = verification.get("diagnostics") or []

        if compiled:
            reward += self.W_COMPILATION
            breakdown["compilation"] = self.W_COMPILATION
        else:
            error_count = sum(1 for d in diag if d.get("level") == "error")
            if error_count == 0 and diag:
                partial = self.W_COMPILATION * 0.5
                reward += partial
                breakdown["compilation"] = partial
            else:
                breakdown["compilation"] = 0.0

        if success:
            reward += self.W_TESTS
            breakdown["tests"] = self.W_TESTS
        elif stage == "complete":
            breakdown["tests"] = self.W_TESTS
        else:
            breakdown["tests"] = 0.0

        unsafe_count = len(re.findall(r'\bunsafe\b', code))
        unsafe_violated = any("unsafe" in c.lower() for c in self._constraints)

        # In transpile phase, we don't heavily penalize unsafe (Neuro-Symbolic step 1)
        # In refactor phase, we DO penalize it (Neuro-Symbolic step 2)
        apply_unsafe_penalty = (self._validation_phase == "refactor")

        if unsafe_count > 0 and unsafe_violated and apply_unsafe_penalty:
            reward -= self.P_UNSAFE_USED
            breakdown["unsafe_penalty"] = -self.P_UNSAFE_USED
            breakdown["memory_safety"] = 0.0
        else:
            total_lines = max(1, len(code.splitlines()))
            safety_score = max(0.0, 1.0 - (unsafe_count / total_lines) * 10)
            mem_reward = round(self.W_MEMORY_SAFE * safety_score, 4)
            reward += mem_reward
            breakdown["memory_safety"] = mem_reward

        cbo = metrics.get("cbo", 0)
        cbo_constraint = any("cbo" in c.lower() for c in self._constraints)

        if cbo_constraint and cbo >= 3:
            reward -= self.P_HIGH_CBO
            breakdown["cbo_penalty"] = -self.P_HIGH_CBO
            breakdown["cbo"] = 0.0
        else:
            cbo_reward = self.W_CBO if cbo < 3 else max(0.0, self.W_CBO * (1 - (cbo - 2) / 5))
            reward += cbo_reward
            breakdown["cbo"] = round(cbo_reward, 4)

        lcom = metrics.get("lcom", 0)
        cohesion_score = max(0.0, 1.0 - lcom / 5.0)
        cohesion_reward = round(self.W_COHESION * cohesion_score, 4)
        reward += cohesion_reward
        breakdown["cohesion"] = cohesion_reward

        clamped = round(max(0.01, min(0.99, reward)), 4)
        breakdown["total"] = clamped
        return clamped, breakdown

    def _format_response(
        self, done: bool, reward: float, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        clamped = round(max(0.01, min(0.99, float(reward))), 4)
        return {
            "observation": self.observation(),
            "reward": clamped,
            "done": done,
            "info": info,
        }
