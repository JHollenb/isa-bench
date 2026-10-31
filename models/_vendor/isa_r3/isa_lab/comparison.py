from __future__ import annotations

from .contracts import PROFILE, TRACK
from .evaluate import ratio


def compare(reports):
    if len(reports) < 2:
        raise ValueError("comparison needs at least two model reports")
    for key in ("schema", "profile", "track", "suite_sha256", "contract_sha256", "split", "oracle", "seed"):
        if len({str(report.get(key)) for report in reports}) != 1:
            raise ValueError(f"comparison identity mismatch: {key}")
    if reports[0]["schema"] != "isa-report-v1" or reports[0]["profile"] != PROFILE or reports[0]["track"] != TRACK:
        raise ValueError("unsupported comparison schema/profile/track")
    if len({report["model"] for report in reports}) != len(reports):
        raise ValueError("duplicate model reports")
    lines = ["# ISA comparison observations", "",
             f"Profile: `{PROFILE}`. Track: `{TRACK}`. Split: `{reports[0]['split']}`.", "",
             f"Oracle: {reports[0]['oracle']}. Suite: `{reports[0]['suite_sha256']}`.", "",
             "| Model | Isolated exact | Rollout step exact | Final state exact | Inference ms | Artifact bytes |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        measured = report["measurement"]
        lines.append(f"| {report['model']} | {ratio(measured['isolated']):.3%} | "
                     f"{ratio(measured['rollout']):.3%} | {ratio(measured['trace_endpoints']):.3%} | "
                     f"{measured['inference_ns'] / 1e6:.3f} | {report['artifact_bytes']} |")
    lines.extend(["", ("Timing is one end-to-end observation, including candidate semantics and commit validation. "
                  "Warmup, backend, machine, order and repeated paired timing must be matched before claiming speed."),
                  ("Artifact bytes count the complete serialized model JSON; they exclude shared executor code, "
                  "Python process memory and training corpus. Native plane bytes and neural parameter counts "
                  "are separate model diagnostics."),
                  ("These are candidate selectors with supplied arithmetic. Full FieldLM and SPFTransformer "
                  "architectures and direct-effect learning require their separately declared tracks."), ""])
    return "\n".join(lines)
