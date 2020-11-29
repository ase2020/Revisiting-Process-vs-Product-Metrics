# Revisiting Process versus Product Metrics: a Large Scale Analysis
Numerous  methods can  build predictive models from software  data. But what methods and conclusions should we endorse as we move from analytics in-the small (dealing with a handful of projects) to analytics in-the-large (dealing with hundreds of projects)? 
To answer this question, we recheck prior small scale results (about process versus product metrics for defect prediction and the granularity of metrics) using 722,471 commits from 700 Github projects. We find that some analytics in-the-small conclusions still hold  when scaling up to analytics in-the large.  For example, like prior work,  we see that  process metrics are    better predictors for defects than product metrics (best process/product-based learners respectively achieve   recalls of 98%/44% and AUCs of 95%/54%, median values). 
That said,  we warn that it is unwise to trust metric importance results from analytics in-the-small studies since those change, dramatically when moving to analytics in-the-large. Also, when reasoning in-the-large about hundreds of projects, it is better to use predictions from multiple models (since single model predictions can become very confused and  exhibit very high variance). 

For the full_paper please refer to the fillowig link - 
https://github.com/ase2020/Revisiting-Process-vs-Product-Metrics/blob/master/Revisiting%20Process%20versus%20Product%20Metrics%20a%20LargeScale%20Analysis.pdf

## This Repository contains the code need to run the RQs of the research paper above. To run the code follow the instruction - 

1) Unzip src.zip, zip files inside data folder and results.zip .
2) Run each python notebooks as marked by the RQs.
3) Some RQs can be run from the same notebook as mentioned in the notebook title.
