We thank reviewers for their insightful comments. We have revised the manuscript to address each comment. Below are our detailed responses:

## Reviewer1 (Revision_Color:Blue)

* **Concern of profiling overhead.**
As discussed in Lines 363-376, our 2D-optimized profiling will not incur extra delay. Instead, it brings *Timely Feedback* and *Better Scheduling Decision* benefits. Our profiler evaluation (Lines 821-827)confirms its capability in large-scale clusters.

* **Applicability to non-DL workloads.**
Similar to prior works [4-11], ROTOM is designed for DL-dedicated clusters (Lines 23-27). It is also applicable to HPC workloads with *iterative patterns*. Extending it to general workloads is a promising direction that serves as our future work (Lines 914-917).

* **More description for Figure 4.**
We add more descriptions about the process of obtaining the branch values (Figure 4) through supervised learning in Lines 562-577.

* **How to handle new jobs?**
We add detailed discussions in &sect;V (Lines 898-908).

* **Why not add intrusion to improve duration prediction?**
We add detailed discussions in &sect;V (Lines 882-897).

* **More clear definition of “non-intrusive”&“interpretable”.**
Great suggestion. We clarify their definitions in Lines 313-322.

* **How many SS values in total?**
Three. We clarify it in Lines 556-558.

* **Compare with RL-based schedulers.**
Thanks for your suggestion. We find it hard to make a **fair** comparison between RL-based schedulers and ROTOM, as they require a tailored environment along with a reward function design to update policy weights, which is not compatible with ROTOM implementation. We cannot find any **open-sourced** code for RL-based schedulers or reproduce prior works due to limited revision time. We will consider this as future work.

* **Figure 7 adjustment.**
Thanks. We make it clearer. 


## Reviewer2 (Revision_Color:Red)

* **More evidence for Section IV.D.(Scalability).**
We add more specific analysis in Lines 845-857.

* **More explicit assumptions of profiling.**
Agree, we clarify assumptions and propose solutions in Lines 405-420.

* **Consider Volcano.**
Indeed, Volcano is a mature system that implements some classic scheduling algorithms (e.g., FIFO). ROTOM can also be integrated into Volcano, which serves as our future work (Lines 917-919).

* **Adjust Tables and Figures.**
Thanks for your suggestion, we make them clearer. 

## Reviewer3 (Revision_Color:Violet)

* **Profiling accuracy when job characteristics may change. & Prior profile may not work since different parameters?**
Great question. Most hyperparameters changes (e.g., `torch.optim.lr_scheduler`) will **not** affect jobs' resource consumption patterns (\(U_G\),\(M_G\),\(U_M\)). To handle such rare corner cases, ROTOM can disable sharing for them (Lines 414-420) and still outperform SOTA algorithm (Figure 8 left).

* **Accuracy of prediction about colocation slowdown? How to collect/measured?**
Following the non-intrusive paradigm, we do not predict or collect colocation slowdown. Instead, we simplify the colocation decision into a classification problem and achieve high accuracy (Table VI, 94.1%) via proactive prediction (Lines 350-355).

* **Compare with EASY backfilling.**
We implement and add evaluation in Table VII and Lines 897-911. ROTOM outperforms optimal EASY (with perfect duration information).

* **Prior work on scheduling.**
Thanks for reference recommendation. We add prior scheduling works related to backfill (Lines 897-900) and duration prediction (Lines 933-944).

* **How is job duration predicted? & Prediction accuracy.**
Detailed duration prediction process is illustrated in Lines 611-624. The prediction accuracy is listed in Table VI (\(R^2\): 0.41) and Lines 877-884 (MAPE: 46.5%).

* **How to profile an application?**
It is feasible and adopted in prior works [5,25,39]. Differently, ROTOM doesn't need to distinguish each iteration intrusively (Lines 183-189).

* **Clarify profile metrics & why high efficient.**
We clarify in &sect;Introduction (Lines 126-131).

* **Why only some models present interference?**
We elaborate interference effect in Lines 258-263. 

* **How to deduce interference effects?**
To validate ROTOM generalizability, we perform comprehensive analysis (Lines 276-284). We collect representative models across different domains (Table I) and plot their mutual colocation effect in Figure 1(d) (over 2000 combinations with different configurations).

* **How adapt to new jobs?**
We add discussion in &sect;V (Lines 898-908).

* **Clarify sharing score scheme.**
We clarify this in Lines 556-576.

* **Why short-term jobs suffer from queuing delays?**
The issue is reported by (SenseTime[6],Microsoft[33],Alibaba[34]), which is incur by runtime-agnostic scheduling algorithm (i.e., FIFO). ROTOM can solve it (Lines 365-368).

* **More details of Affine-jobpair Binder. & How to determine jobs not likely to cause interference?**
We clarify it in Lines 423-445. We adopt Indolent Packing that **inactively** colocates jobs and adds more constraints to improve the confidence. The effect is clearly presented (Lines 546-556): [**Figure 4**] Tiny (most \(U_G<25\%\)) and Medium (most \(U_G<53\%\)) jobs are colocatable; [**Figure 1(d)**] over 0.95x speed (accumulative \(U_G<50%\)) and over 0.8x speed (accumulative \(U_G<100%\)). 

* **Is profiling cluster different from the main cluster?**
No. They share the same physical machines, but are logically partitioned with resource quota guarantee.

* **Are the job estimates done in exclusive mode?**
Yes (Lines 613-614).

* **How were the thresholds selected in Figure 4?**
Automatically generated by the supervised learning model. More details in Lines 561-577.

* **Clarify features in III-C.**
We add clarification in Lines 546-549.

* **Clarify extract trend.**
We elaborate in Lines 588-592.

* **How to predict future cluster utilization?**
Based on Throughput Predict Model (Lines 631-637).

* **Evaluated job numbers.**
We summarized them in Table II.

* **More explanations to figures and tables.**
We add more explanations (Tables II,IV; Figures 3,4).

* **Clarify *non-intrusive scheduler*.**
We clarify it in Lines 313-316.

* **Clarify *2D job profiler*.**
We clarify it in Lines 378-381.

* **Organization and content refinements.**
Thanks, we polish the paper by (1)Fixed typos and inaccurate words (Line 248,402); (2)Add citation (Line 292); (3)More refer to Figure 3 before introducing each component (Lines 345,425,465,497,640,670). 


## Reviewer4 (Revision_Color:Teal)

* **More empirical evaluation.**
Thanks, we add additional evaluations on a dynamic submission trace with more jobs and longer average duration (Table III and Lines 731-748). We further consider Tiresias (QSSF and Horus didn't release their implementation).

* **Explaining results under different scheduling approaches.**
Great suggestion. We add more introductions and explanations in Lines 688-712 and 745-750.

* **Compare to AntMan & Gavel.**
We didn't serve them as baselines because: (1) AntMan is an Alibaba-internal scheduling system (not open-sourced). It intrudes Tensorflow&Pytorch and is hard to implement. (2) Gavel is a mechanism that enhances existing scheduling policies to support ***GPU heterogeneity*** consideration. However, our evaluation is based on a ***homogenous*** setting thus Gavel becomes invalid. Our prototype leverages Gavel's *gRPC* implementation instead of its' scheduling mechanism (Lines 621-624).

 




--------
# Artifact Evaluation

Dear Committee Members,

Thanks for your valuable comments on the appendix. We have carefully revised our appendix and open-source artifact. Here are our responses to each comment.

1. (For Reviewer 1) After censored by our organization, we released all the trace data and framework scripts for reproducing QSSF and CES services in our paper.

2. (For Reviewer 1 & 3) We added a step-by-step introduction in the README file to reproduce results.

3. (For Reviewer 2 & 3) We added detailed hardware descriptions on datacenter and our analysis platform, including CPU, GPU, memory, and network information.

4. (For Reviewer 2) Additionally, we have applied all badges as you suggested.

Best Regards,
Anonymous Authors