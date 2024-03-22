# [Event-based eye Tracking Challenge Submission](https://EETChallenge.github.io/EET.github.io/) \ Using an efficient version of Recurrent Vision Transformer

This is our submission to the Event-based eye Tracking challenge part of the CVPR 2024 Workshop [AI for Streaming](https://ai4streaming-workshop.github.io/).


## About the Challenge
Developing an event-based eye-tracking system presents significant opportunities in diverse fields, notably in consumer electronics and neuroscience. Human eyes exhibit rapid movements, occasionally surpassing speeds of 300°/s. This necessitates using [event cameras](https://www.youtube.com/watch?v=6xOmo7Ikwzk&t=80s&ab_channel=Sony-Global) capable of high-speed sampling and tracking. 

In consumer electronics, particularly in augmented and virtual reality (AR/VR) applications, the primary benefits of event-based systems extend beyond their high speed. Their highly sparse input data streams can be exploited to reduce power consumption. This is a pivotal advantage in creating lighter, more efficient wearable headsets that offer prolonged usage and enhanced user comfort. 

This is instrumental in augmenting the immersive experience in AR/VR and expanding the capabilities of portable technology. In neuroscience and cognitive studies, such technology is crucial for deciphering the complexities of eye movement. It facilitates a deeper comprehension of visual attention processes and aids in diagnosing and understanding neurological disorders. 

This challenge aims to develop an **event-based eye-tracking system for precise tracking of rapid eye movements** to produce lighter and more comfortable devices for a better user experience. Simultaneously, it promises to provide novel insights into neuroscience and cognitive research, deepening our understanding of these domains.

## Our solution
We have been inspired by the Recurrent Vision Transformer model that demonstrated great performance in event-based object detection. Starting from the original architecture we have made modifications in order to better fit our task and performance requirements.

<img src="figures/b.svg">



## References

```
@inproceedings{chen20233et,
  title={3et: Efficient Event-based Eye Tracking Using a Change-based Convlstm Network},
  author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
  booktitle={2023 IEEE Biomedical Circuits and Systems Conference (BioCAS)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```