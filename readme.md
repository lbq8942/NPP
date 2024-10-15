The source code of paper "Cumulative Hazard Function Based Efficient Multivariate Temporal Point Process Learning" (IJCNN2024).

### Description

Most existing temporal point process models are characterized by conditional intensity function, which often require numerical approximation methods for likelihood evaluation and thus potentially hurts their performance. By directly modelling the integral of the intensity function, i.e., the cumulative hazard function (CHF), the likelihood can be evaluated accurately, making it a promising approach. However, existing CHF-based methods are not well-defined, i.e., the mathematical constraints of CHF are not completely satisfied, leading to untrustworthy results. For multivariate temporal point process, most existing methods model intensity (or density, etc.) functions for each variate, limiting the scalability. 

In this paper, we explore using neural networks to model a flexible but well-defined CHF and learning the multivariate temporal point process with low parameter complexity. 

### Model:EFullyNN

![efullynn](https://github.com/user-attachments/assets/bd5c3f88-2200-485d-b44f-4332dff3fd80)

### Running
Before running the following command, modify the parameter "pro_path" in *args.py* as your *NPP* path.

```powershell
python main.py   --data hawkes_deer1  --baseline EFullyNN --data_dir EFullyNN --gpu 1 
```

## Citation
```
@INPROCEEDINGS{CHFliu,
  author={Liu, Bingqing},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Cumulative Hazard Function Based Efficient Multivariate Temporal Point Process Learning}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Limiting;Source coding;Scalability;Neural networks;Memory management;Predictive models;Hazards;temporal point process;cumulative hazard function;event sequence modelling},
  doi={10.1109/IJCNN60899.2024.10650460}}
}
```






