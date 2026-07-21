# Paper-to-Code Alignment

## D-EMA: equations (1)-(11)

| Manuscript operation | Implementation |
|---|---|
| Dynamic group count `G=min(Gmax,floor(C/Tc))` | `D_EMA.__init__` |
| Horizontal and vertical pooling | `pool_h`, `pool_w` |
| Spatial fusion and normalization | `spatial_fuse`, sigmoid, 2-D softmax |
| Local context `Conv3x3 + BN + ReLU` | `local_conv`, `local_bn`, `local_act` |
| `F_ref = F_local ⊙ A_spatial` | `refined = local_feature * spatial_attention` |
| `F_enh = F_ref + alpha*x` | `enhanced = refined + self.alpha * grouped` |
| Channel reassembly | final reshape |

The manuscript's directional descriptors have size `H+W`, while its final attention is specified as `1xHxW`. The code converts the horizontal and vertical descriptors into a two-dimensional map by additive outer fusion before the spatial softmax. This explicit step resolves the dimensional transition that is not fully specified in the text.

## AQFL: equations (12)-(18)

| Equation | Implementation |
|---|---|
| (12) effective probability | `p_t` |
| (13) continuous alpha interpolation | `alpha_t` |
| (14) dynamic focusing factor | `gamma_i` |
| (15) aligned IoU | `aligned_box_iou` |
| (16) quality weight | `quality_weight` |
| (17) per-sample loss | `formulation="paper"` |
| (18) aggregation | `reduction` or YOLO normalization |

For a continuous target, `BCE(logit,y)` is not equal to `-log(p_t)`. Therefore the exact manuscript path is the default. The legacy BCE interpretation is explicitly named `soft_bce`.

## QHA-NMS: equations (19)-(24)

| Manuscript operation | Implementation |
|---|---|
| (19) local density | `_density_and_agreement` |
| density normalization `D_i` | mean same-class overlap |
| (20) per-box threshold | `density_adaptive_thresholds` |
| (21) Stage-I adaptive NMS | first `heterogeneous_greedy_nms` call |
| (22) quality vector | confidence, localization quality, boundary quality |
| (23) quality fusion | weighted sum in `qha_nms_boxes` |
| (24) aspect-ratio penalty | `boundary_quality` |
| Stage-II ranking/suppression | second `heterogeneous_greedy_nms` call |

### Explicit engineering choices

1. **Pairwise threshold rule.** The manuscript defines `T_i` but does not state whether a pair uses the selected threshold, candidate threshold, minimum, or mean. The default is `min(T_i,T_j)` and is configurable.
2. **Localization quality at inference.** Ground-truth IoU is unavailable during inference. A predicted IoU/quality head is preferred. Without one, the code uses same-class prediction agreement as a proxy.
3. **Boundary formula.** Absolute log-ratio deviation is used so tall and wide reciprocal deviations receive symmetric penalties and the result remains bounded.
4. **Complexity.** Density computation is chunked and candidates are confidence-pruned before QHA-NMS.
