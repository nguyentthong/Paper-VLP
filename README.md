# Papers of Vision-and-Language Pretraining (VLP)

- [Papers of Vision-and-Language Pretraining (VLP)](#papers-of-vision-and-language-pretraining-vlp)
    - [VLP with Different Motivations](#vlp-with-different-motivations)
        - [First Moves for VLP](#first-moves-for-vlp)
        - [Diversification of VLP Approaches](#diversification-of-vlp-approaches)
        - [Polishing Representations in VLP](#polishing-representations-in-vlp)
        - [End2End VLP](#end2end-vlp)
        - [VLP Applications](#vlp-applications)
        - [Assessment of Risks in VLP](#assessment-of-risks-in-vlp)
        - [Model Compression for VLP](#model-compression-for-vlp)
        - [Multilinguality in VLP](#multilinguality-in-vlp)
        - [Probing Analysis in VLP](#probing-analysis-in-vlp)
        - [Datasets for VLP](#datasets-for-vlp)
    - [Future Directions and Prospective Issues](#future-directions-and-prospective-issues)
        - [Enhancing Interpretability of VLP Models](#enhancing-interpretability-of-vlp-models)
        - [Evaluation of VLP Models](#evaluation-of-vlp-models)
        - [Practical Implementation of VLP Models](#practical-implementation-of-vlp-models)
    - [Survey Survey of Vision-and-Language Models](#survey-of-vision-and-language-models)

## VLP with Different Motivations

### First Moves for VLP
1. **Dialog-based interactive image retrieval** *Xiaoxiao Guo, Hui Wu, Yu Cheng, Steven Rennie, Gerald Tesauro, Rogerio Schmidt Feris* `NeurIPS 2018` [[pdf]](https://arxiv.org/abs/1805.00145) [[code]](https://github.com/XiaoxiaoGuo/fashion-retrieval)

1. **Bilinear attention networks** *Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang* `NeurIPS 2018` [[pdf]](https://arxiv.org/abs/1805.07932) [[code]](https://github.com/jnhwkim/ban-vqa)

1. **Advancing state-of-the-art image recognition with deep learning on hashtags** *Manohar Paluri, Dhruv Mahajan, Ross Girshick, Vignesh Ramanathan* [[pdf]](https://engineering.fb.com/2018/05/02/ml-applications/advancing-state-of-the-art-image-recognition-with-deep-learning-on-hashtags/) 

1. **Bottom-up and top-down attention for image captioning and visual question answering** *Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, Lei Zhang* `CVPR 2018` [[pdf]](https://arxiv.org/abs/1707.07998) [[code]](https://github.com/peteanderson80/bottom-up-attention)

1. **Relation-aware graph attention network for visual question answering** *Linjie Li, Zhe Gan, Yu Cheng, Jingjing Liu* `ICCV 2019` [[pdf]](https://arxiv.org/abs/1903.12314) [[code]](https://github.com/linjieli222/VQA_ReGAT)

1. **Lxmert: Learning cross-modality encoder representations from transformers** *Hao Tan, Mohit Bansal* `EMNLP 2019` [[pdf]](https://arxiv.org/abs/1908.07490) [[code]](https://github.com/airsplay/lxmert)

1. **Spatio-temporal dynamics and semantic attribute enriched visual encoding for video captioning** *Nayyer Aafaq, Naveed Akhtar, Wei Liu, Syed Zulqarnain Gilani, Ajmal Mian* `CVPR 2019` [[pdf]](https://arxiv.org/abs/1902.10322)

1. **Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks** *Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1908.02265) [[code]](https://github.com/facebookresearch/vilbert-multi-task)

1. **Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training** *Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, Ming Zhou* `AAAI 2020` [[pdf]](https://arxiv.org/abs/1908.06066)

1. **Uniter: Universal image-text representation learning** *Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu* `ECCV 2020` [[pdf]](https://arxiv.org/abs/1909.11740) [[code]](https://github.com/ChenRocks/UNITER)

1. **Unified vision-language pre-training for image captioning and vqa** *Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J. Corso, Jianfeng Gao* `AAAI 2020` [[pdf]](https://arxiv.org/abs/1909.11059) [[code]](https://github.com/LuoweiZhou/VLP)

1. **Visualbert: A simple and performant baseline for vision and language** *Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1908.03557) [[code]](https://github.com/uclanlp/visualbert)

1. **Vl-bert: Pre-training of generic visual-linguistic representations** *Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai* `ICLR 2020` [[pdf]](https://arxiv.org/abs/1908.08530) [[code]](https://github.com/jackroos/VL-BERT)

1. **Fusion of detected objects in text for visual question answering** *Chris Alberti, Jeffrey Ling, Michael Collins, David Reitter* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1908.05054) [[code]](https://github.com/google-research/language)

1. **X-lxmert: Paint, caption and answer questions with multi-modal transformers** *Jaemin Cho, Jiasen Lu, Dustin Schwenk, Hannaneh Hajishirzi, Aniruddha Kembhavi* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2009.11278) [[code]](https://github.com/allenai/x-lxmert)

1. **Iterative answer prediction with pointeraugmented multimodal transformers for textvqa** *Ronghang Hu, Amanpreet Singh, Trevor Darrell, Marcus Rohrbach* `CVPR 2020` [[pdf]](https://arxiv.org/abs/1911.06258) [[code]](https://github.com/adlnlp/attention_vl)

1. **Mural: multimodal, multitask retrieval across languages** *Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, Jason Baldridge* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2109.05125) 

1. **Align before fuse: Vision and language representation learning with momentum distillation** *Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, Steven Hoi* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2107.07651) [[code]](https://github.com/salesforce/lavis)

1. **Unimo: Towards unified-modal understanding and generation via cross-modal contrastive learning** *Wei Li, Can Gao, Guocheng Niu, Xinyan Xiao, Hao Liu, Jiachen Liu, Hua Wu, Haifeng Wang* `ACL 2021` [[pdf]](https://arxiv.org/abs/2012.15409) [[code]](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_generation/unimo-text)

1. **Interbert: Vision-and-language interaction for multi-modal pretraining** *Junyang Lin, An Yang, Yichang Zhang, Jie Liu, Jingren Zhou, Hongxia Yang* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2003.13198)

1. **Simvlm: Simple visual language model pretraining with weak supervision** *Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, Yuan Cao* `ICLR 2022` [[pdf]](https://arxiv.org/abs/2108.10904)

1. **Xgpt: Cross-modal generative pre-training for image captioning** *Qiaolin Xia, Haoyang Huang, Nan Duan, Dongdong Zhang, Lei Ji, Zhifang Sui, Edward Cui, Taroon Bharti, Xin Liu, Ming Zhou* `NLPCC 2021` [[pdf]](https://arxiv.org/abs/2003.01473)

### Diversification of VLP Approaches

1. **Unifying vision-and-language tasks via text generation** *Jaemin Cho, Jie Lei, Hao Tan, Mohit Bansal* `ICML 2021` [[pdf]](https://arxiv.org/abs/2102.02779) [[code]](https://github.com/j-min/VL-T5)

1. **Lightningdot: Pre-training visual-semantic embeddings for real-time image-text retrieval** *Siqi Sun, Yen-Chun Chen, Linjie Li, Shuohang Wang, Yuwei Fang, Jingjing Liu* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2103.08784) [[code]](https://github.com/intersun/LightningDOT)

1. **Unsupervised vision-and-language pre-training without parallel images and captions** *Liunian Harold Li, Haoxuan You, Zhecan Wang, Alireza Zareian, Shih-Fu Chang, Kai-Wei Chang* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2010.12831) [[code]](https://github.com/uclanlp/visualbert)

1. **12-in-1: Multi-task vision and language representation learning** *Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, Stefan Lee* `CVPR 2020` [[pdf]](https://arxiv.org/abs/1912.02315) [[code]](https://github.com/facebookresearch/vilbert-multi-task)

1. **Ernie-vil: Knowledge enhanced vision-language representations through scene graph** *Fei Yu, Jiji Tang, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, Haifeng Wang* `AAAI 2021` [[pdf]](https://arxiv.org/abs/2006.16934)

1. **Large-scale adversarial training for vision-and-language representation learning** *Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, Jingjing Liu* `NeurIPS 2020` [[pdf]](https://arxiv.org/abs/2006.06195) [[code_1]](https://github.com/zhegan27/VILLA) [[code_2]](https://github.com/zhegan27/LXMERT-AdvTrain)

1. **Contrastive visual-linguistic pretraining** *Lei Shi, Kai Shuang, Shijie Geng, Peng Su, Zhengkai Jiang, Peng Gao, Zuohui Fu, Gerard de Melo, Sen Su* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2007.13135) [[code]](https://github.com/ArcherYunDong/CVLP-)

1. **Lamp: label augmented multimodal pretraining** *Jia Guo, Chen Zhu, Yilun Zhao, Heda Wang, Yao Hu, Xiaofei He, Deng Cai* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2012.04446)

### Polishing Representations in VLP

1. **Vinvl: Revisiting visual representations in vision-language models** *Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, Jianfeng Gao* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2101.00529) [[code]](https://github.com/pzzhang/VinVL)

1. **Oscar: Object-semantics aligned pre-training for vision-language tasks** *Xiujun Li, Xi Yin, Chunyuan Li, Pengchuan Zhang, Xiaowei Hu, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, Yejin Choi, Jianfeng Gao* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2004.06165) [[code]](https://github.com/microsoft/Oscar)

1. **Learning visual representations with caption annotations** *Mert Bulent Sariyildiz, Julien Perez, Diane Larlus* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2008.01392)

1. **Virtex: Learning visual representations from textual annotations** *Karan Desai, Justin Johnson* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2006.06666) [[code]](https://github.com/kdexd/virtex)

1. **Vokenization: Improving language understanding with contextualized, visual-grounded supervision** *Hao Tan, Mohit Bansal* `EMNLP 2020` [[pdf]](https://arxiv.org/abs/2010.06775) [[code]](https://github.com/airsplay/vokenization)

1. **Scaling up visual and vision-language representation learning with noisy text supervision** *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig* `ICML 2021` [[pdf]](https://arxiv.org/abs/2102.05918) [[code]](https://github.com/kakaobrain/coyo-dataset)

1. **Learning transferable visual models from natural language supervision** *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever* `ICML 2021` [[pdf]](https://arxiv.org/abs/2103.00020) [[code]](https://github.com/leaderj1001/CLIP)

1. **Vlmo: Unified vision-language pre-training with mixture-of-modality-experts** *Hangbo Bao, Wenhui Wang, Li Dong, Qiang Liu, Owais Khan Mohammed, Kriti Aggarwal, Subhojit Som, Furu Wei* `NeurIPS 2022` [[pdf]](https://arxiv.org/abs/2111.02358) [[code]](https://github.com/microsoft/unilm/tree/master/vlmo)

1. **Data efficient masked language modeling for vision and language** *Yonatan Bitton, Gabriel Stanovsky, Michael Elhadad, Roy Schwartz* `EMNLP 2021 findings` [[pdf]](https://arxiv.org/abs/2109.02040) [[code]](https://github.com/yonatanbitton/data_efficient_masked_language_modeling_for_vision_and_language)

1. **Flava: A foundational language and vision alignment model** *Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, Douwe Kiela* `CVPR 2022` [[pdf]](https://arxiv.org/abs/2112.04482) [[code]](https://flava-model.github.io/)

1. **Multimodal few-shot learning with frozen language models** *Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill* `NeurIPS 2022` [[pdf]](https://arxiv.org/abs/2106.13884) [[code]](https://github.com/ilkerkesen/frozen)

### End2End VLP

1. **Pixel-bert: Aligning image pixels with text by deep multi-modal transformers** *Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, Jianlong Fu* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2004.00849) [[code]](https://github.com/microsoft/xpretrain)

1. **Vilt: Vision-and-language transformer without convolution or region supervision** *Wonjae Kim, Bokyung Son, Ildoo Kim* `ICML 2021` [[pdf]](https://arxiv.org/abs/2102.03334) [[code]](https://github.com/dandelin/vilt)

1. **Seeing out of the box: End-to-end pre-training for vision-language representation learning** *Zhicheng Huang, Zhaoyang Zeng, Yupan Huang, Bei Liu, Dongmei Fu, Jianlong Fu* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2104.03135) [[code]](https://github.com/researchmm/soho)

1. **E2e-vlp: End-to-end vision-language pre-training enhanced by visual learning** *Haiyang Xu, Ming Yan, Chenliang Li, Bin Bi, Songfang Huang, Wenming Xiao, Fei Huang* `ACL 2021` [[pdf]](https://arxiv.org/abs/2106.01804)

1. **An image is worth 16x16 words: Transformers for image recognition at scale** *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby* `ICLR 2020` [[pdf]](https://arxiv.org/abs/2010.11929) [[code]](https://github.com/google-research/vision_transformer)

### VLP Applications

1. **Vd-bert: A unified vision and dialog transformer with bert** *Yue Wang, Shafiq Joty, Michael R. Lyu, Irwin King, Caiming Xiong, Steven C.H. Hoi* `EMNLP2020` [[pdf]](https://arxiv.org/abs/2004.13278) [[code]](https://github.com/salesforce/VD-BERT)

1. **Towards learning a generic agent for vision-and-language navigation via pre-training** *Weituo Hao, Chunyuan Li, Xiujun Li, Lawrence Carin, Jianfeng Gao* `CVPR 2020` [[pdf]](https://arxiv.org/abs/2002.10638) [[code]](https://github.com/weituo12321/PREVALENT)

1. **Vivo: Visual vocabulary pre-training for novel object captioning** *Xiaowei Hu, Xi Yin, Kevin Lin, Lijuan Wang, Lei Zhang, Jianfeng Gao, Zicheng Liu* `AAAI 2021` [[pdf]](https://arxiv.org/abs/2009.13682)

1. **Tap: Text-aware pre-training for text-vqa and text-caption** *Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florencio, Lijuan Wang, Cha Zhang, Lei Zhang, Jiebo Luo* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2012.04638) [[code]](https://github.com/microsoft/TAP)

1. **Kaleido-bert: Vision-language pre-training on fashion domain** *Mingchen Zhuge, Dehong Gao, Deng-Ping Fan, Linbo Jin, Ben Chen, Haoming Zhou, Minghui Qiu, Ling Shao* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2103.16110) [[code]](https://github.com/mczhuge/Kaleido-BERT)

1. **Large-scale pretraining for visual dialog: A simple state-of-the-art baseline** *Vishvak Murahari, Dhruv Batra, Devi Parikh, Abhishek Das* `ECCV 2020` [[pdf]](https://arxiv.org/abs/1912.02379) [[code]](https://github.com/vmurahari3/visdial-bert)

1. **Reasoning over vision and language: Exploring the benefits of supplemental knowledge** *Violetta Shevchenko, Damien Teney, Anthony Dick, Anton van den Hengel* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2101.06013)

1. **Multimodal review generation for recommender systems** *Quoc-Tuan Truong, Hady Lauw* `WWW 2019` [[pdf]](https://dl.acm.org/doi/fullHtml/10.1145/3308558.3313463)

1. **Vln bert: A recurrent vision-and-language bert for navigation** *Yicong Hong, Qi Wu, Yuankai Qi, Cristian Rodriguez-Opazo, Stephen Gould* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2011.13922) [[code]](https://github.com/YicongHong/Recurrent-VLN-BERT)

1. **Curriculum learning for vision-and-language navigation** *Jiwen Zhang, Zhongyu Wei, Jianqing Fan, Jiajie Peng* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2111.07228) [[code]](https://github.com/IMNearth/Curriculum-Learning-For-VLN)

1. **Show, attend and tell: Neural image caption generation with visual attention** *Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio* `ICML 2015` [[pdf]](https://arxiv.org/abs/1502.03044) [[code]](https://github.com/djain454/Show-Attend-and-Tell-Neural-Image-Caption-Generation-with-Visual-Attention)

1. **End-to-end object detection with transformers** *Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2005.12872) [[code]](https://github.com/facebookresearch/detr)

1. **Egnet: Edge guidance network for salient object detection** *Jia-Xing Zhao, Jiangjiang Liu, Den-Ping Fan, Yang Cao, Jufeng Yang, Ming-Ming Cheng* `ICCV 2019` [[pdf]](https://arxiv.org/abs/1908.08297) [[code]](https://github.com/JXingZhao/EGNet)

1. **Fashionbert: Text and image matching with adaptive loss for cross-modal retrieval** *Dehong Gao, Linbo Jin, Ben Chen, Minghui Qiu, Peng Li, Yi Wei, Yi Hu, Hao Wang* `SIGIR 2020` [[pdf]](https://arxiv.org/abs/2005.09801)

### Assessment of Risks in VLP

1. **A closer look at the robustness of vision-and-language pre-trained models** *Linjie Li, Zhe Gan, Jingjing Liu* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2012.08673)

1. **Causal attention for vision-language tasks** *Xu Yang, Hanwang Zhang, Guojun Qi, Jianfei Cai* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2103.03493) [[code]](https://github.com/yangxuntu/lxmertcatt)

1. **Adversarial vqa: A new benchmark for evaluating the robustness of vqa models** *Linjie Li, Jie Lei, Zhe Gan, Jingjing Liu* `ICCV 2021` [[pdf]](https://arxiv.org/abs/2106.00245)

1. **Worst of both worlds: Biases compound in pre-trained vision-and-language models** *Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, Douwe Kiela* `CVPR 2022` [[pdf]](https://arxiv.org/abs/2112.04482)

1. **Attacking visual language grounding with adversarial examples: A case study on neural image captioning** *Hongge Chen, Huan Zhang, Pin-Yu Chen, Jinfeng Yi, Cho-Jui Hsieh* `ACL 2018` [[pdf]](https://arxiv.org/abs/1712.02051) [[code]](https://github.com/huanzhang12/ImageCaptioningAttack)

### Model Compression for VLP

1. **Minivlm: A smaller and faster vision-language model** *Jianfeng Wang, Xiaowei Hu, Pengchuan Zhang, Xiujun Li, Lijuan Wang, Lei Zhang, Jianfeng Gao, Zicheng Liu* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2012.06946)

1. **Compressing visual-linguistic model via knowledge distillation** *Zhiyuan Fang, Jianfeng Wang, Xiaowei Hu, Lijuan Wang, Yezhou Yang, Zicheng Liu* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2104.02096)

1. **Playing lottery tickets with vision and language** *Zhe Gan, Yen-Chun Chen, Linjie Li, Tianlong Chen, Yu Cheng, Shuohang Wang, Jingjing Liu, Lijuan Wang, Zicheng Liu* `AAAI 2022` [[pdf]](https://arxiv.org/abs/2104.11832)

### Multilinguality in VLP

1. **Uc2: Universal cross-lingual cross-modal vision-and-language pre-training** *Mingyang Zhou, Luowei Zhou, Shuohang Wang, Yu Cheng, Linjie Li, Zhou Yu, Jingjing Liu* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2104.00332) [[code]](https://github.com/zmykevin/UC2)

1. **M3p: Learning universal representations via multitask multilingual multimodal pre-training** *Minheng Ni, Haoyang Huang, Lin Su, Edward Cui, Taroon Bharti, Lijuan Wang, Jianfeng Gao, Dongdong Zhang, Nan Duan* `CVPR 2021` [[pdf]](https://arxiv.org/abs/2006.02635) [[code]](https://github.com/microsoft/M3P)

1. **Multilingual multimodal pre-training for zero-shot cross-lingual transfer of vision-language models** *Po-Yao Huang, Mandela Patrick, Junjie Hu, Graham Neubig, Florian Metze, Alexander Hauptmann* `NAACL 2021` [[pdf]](https://arxiv.org/abs/2103.08849) [[code]](https://github.com/berniebear/Multi-HT100M)

### Probing Analysis in VLP

1. **Behind the scene: Revealing the secrets of pretrained vision-and-language models** *Jize Cao, Zhe Gan, Yu Cheng, Licheng Yu, Yen-Chun Chen, Jingjing Liu* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2005.07310)

1. **What does bert with vision look at?** *Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, Kai-Wei Chang* `ACL 2020` [[pdf]](https://aclanthology.org/2020.acl-main.469/)

1. **Decoupling the role of data, attention, and losses in multimodal transformers** *Lisa Anne Hendricks, John Mellor, Rosalia Schneider, Jean-Baptiste Alayrac, Aida Nematzadeh* `MIT Press Publication 2021` [[pdf]](https://arxiv.org/abs/2102.00529) [[code]](https://github.com/deepmind/multimodal_transformers)

1. **Probing inter-modality: Visual parsing with self-attention for vision-and-language pre-training** *Hongwei Xue, Yupan Huang, Bei Liu, Houwen Peng, Jianlong Fu, Houqiang Li, Jiebo Luo* `NeurIPS 2021` [[pdf]](https://arxiv.org/abs/2106.13488)

1. **Analyzing compositionality in visual question answering** *Sanjay Subramanian, Sameer Singh, Matt Gardner* `ViGIL@NeurIPS 2019` [[pdf]](https://aclanthology.org/D16-1203.pdf)

1. **Are we pretraining it right? Digging deeper into visio-linguistic pretraining** *Amanpreet Singh, Vedanuj Goswami, Devi Parikh* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2004.08744)

### Datasets for VLP

1. **The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale** *Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, Vittorio Ferrari* `International Journal of Computer Vision 2020` [[pdf]](https://arxiv.org/abs/1811.00982) [[code]](https://github.com/ccc013/DeepLearning_Notes)

1. **Textcaps: a dataset for image captioning with reading comprehension** *Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, Amanpreet Singh* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2003.12462)

1. **Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation** *Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2201.12086) [[code]](https://github.com/salesforce/lavis)

1. **Zero-shot text-to-image generation** *Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever* `PMLR 2021` [[pdf]](https://arxiv.org/abs/2102.12092)

## Future Directions and Prospective Issues

### Enhancing Interpretability of VLP Models

1. **Is attention interpretable?** *Sofia Serrano, Noah A. Smith* `ACL 2019` [[pdf]](https://arxiv.org/abs/1906.03731)

1. **Interpretable deep learning: Interpretation, interpretability, trustworthiness, and beyond** *Xuhong Li, Haoyi Xiong, Xingjian Li, Xuanyu Wu, Xiao Zhang, Ji Liu, Jiang Bian, Dejing Dou* `arXiv 2021` [[pdf]](https://arxiv.org/abs/2103.10689) [[code]](https://github.com/PaddlePaddle/InterpretDL)

1. **Interpretable convolutional neural networks with dual local and global attention for review rating prediction** *Sungyong Seo, Jing Huang, Hao Yang, Yan Liu* `ACM Conference on Recommender Systems 2017` [[pdf]](https://dl.acm.org/doi/10.1145/3109859.3109890) [[code]](https://github.com/seongjunyun/CNN-with-Dual-Local-and-Global-Attention)

### Evaluation of VLP Models

1. **Vqa-lol: Visual question answering under the lens of logic** *Tejas Gokhale, Pratyay Banerjee, Chitta Baral, Yezhou Yang* `ECCV 2020` [[pdf]](https://arxiv.org/abs/2002.08325)

1. **Towards causal vqa: Revealing and reducing spurious correlations by invariant and covariant semantic editing** *Vedika Agarwal, Rakshith Shetty, Mario Fritz* `arXiv 2019`[[pdf]](https://arxiv.org/abs/1912.07538)

1. **Roses are red, violets are blue... but should vqa expect them to?** *Corentin Kervadec, Grigory Antipov, Moez Baccouche, Christian Wolf* `arXiv 2020` [[pdf]](https://arxiv.org/abs/2006.05121) [[code]](https://github.com/gqa-ood/gqa-ood)

## Survey of Vision-and-Language Models

1. **Vlp: A survey on vision-language pre-training** *Feilong Chen, Duzhen Zhang, Minglun Han, Xiuyi Chen, Jing Shi, Shuang Xu, Bo Xu* `arXiv 2022` [[pdf]](https://arxiv.org/abs/2202.09061) [[code]](https://github.com/phellonchen/awesome-Vision-and-Language-Pre-training)

1. **A survey of vision-language pre-trained models** *Yifan Du, Zikang Liu, Junyi Li, Wayne Xin Zhao* `IJCAI 2022` [[pdf]](https://arxiv.org/abs/2202.10936)

1. **A survey on automatic image caption generation** *Shuang Bai, Shan An* `Neurocomputing 2018` [[pdf]](https://www.sciencedirect.com/science/article/abs/pii/S0925231218306659)

1. **Automatic description generation from images: A survey of models, datasets, and evaluation measures** *Raffaella Bernardi, Ruket Cakici, Desmond Elliott, Aykut Erdem, Erkut Erdem, Nazli Ikizler-Cinbis, Frank Keller, Adrian Muscat, Barbara Plank* `Journal of Artificial Intelligence Review 2016`[[pdf]](https://arxiv.org/abs/1601.03896)

1. **Deep multimodal representation learning: A survey** *Wenzhong Guo, Jianwen Wang, Shiping Wang* `IEEE Access 2019` [[pdf]](https://ieeexplore.ieee.org/document/8715409)

1. **A comprehensive survey of deep learning for image captioning** *Md. Zakir Hossain, Ferdous Sohel, Mohd Fairuz Shiratuddin, Hamid Laga* `ACM Computing Surveys 2018` [[pdf]](https://arxiv.org/abs/1810.04020) [[code]](https://github.com/HemanthTejaY/Deep-Learning-Image-Captioning---A-comparitive-study)

1. **Video question-answering techniques, benchmark datasets and evaluation metrics leveraging video captioning: A comprehensive survey** *Khushboo Khurana, Umesh Deshpande* `IEEE Access 2021` [[pdf]](https://ieeexplore.ieee.org/document/9350580)

1. **Visual to text: Survey of image and video captioning** *Sheng Li, Zhiqiang Tao, Kang Li, Yun Fu* `IEEE Transactions on Emerging Topics in Computational Intelligence 2019` [[pdf]](https://ieeexplore.ieee.org/ielaam/7433297/8768200/8627985-aam.pdf)

1. **From show to tell: A survey on image captioning** *Matteo Stefanini, Marcella Cornia, Lorenzo Baraldi, Silvia Cascianelli, Giuseppe Fiameni, Rita Cucchiara* `IEEE Transactions on Pattern Analysis and Machine Intelligence 2023` [[pdf]](https://arxiv.org/abs/2107.06912)

1. **Visual question answering: A survey of methods and datasets** *Qi Wu, Damien Teney, Peng Wang, Chunhua Shen, Anthony Dick, Anton van den Hengel* `CVIU 2017` [[pdf]](https://arxiv.org/abs/1607.05910)

1. **Bridging vision and language from the video-to-text perspective: A comprehensive review** *Jesus Perez-Martin, Benjamin Bustos, Silvio Jamil F. Guimarães, Ivan Sipiran, Jorge Pérez, Grethel Coello Said* `Artificial Intelligence Review 2021` [[pdf]](https://arxiv.org/abs/2103.14785)

1. **Visual question answering: Datasets, algorithms, and future challenges** *Kushal Kafle, Christopher Kanan* `arXiv 2016` [[pdf]](https://arxiv.org/abs/1610.01465) [[code]](https://github.com/andreiluca96/reversed-vqa)

1. **Challenges and prospects in vision and language research** *Kushal Kafle, Robik Shrestha, Christopher Kanan* `arXiv 2019` [[pdf]](https://arxiv.org/abs/1904.09317)

1. **Multimodal intelligence: Representation learning, information fusion, and applications** *Chao Zhang, Zichao Yang, Xiaodong He, Li Deng* `IEEE Journal of Selected Topics in Signal Processing 2020` [[pdf]](https://arxiv.org/abs/1911.03977) 