# LEMURIA — Language Emergence and Metrics for Understanding Relayed Information across Agents

This GitHub contains the code for our toolkit LEMURIA for training and evaluating emergent communication between agents.

### Citation

If this repository was useful for your research, please consider citing our publication:

```bibtex
@inproceedings{bernard-mickus-2023-many,
    title = "So many design choices: Improving and interpreting neural agent communication in signaling games",
    author = "Bernard, Timoth{\'e}e  and
      Mickus, Timothee",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.531",
    doi = "10.18653/v1/2023.findings-acl.531",
    pages = "8399--8413",
    abstract = "Emergent language games are experimental protocols designed to model how communication may arise among a group of agents. In this paper, we focus on how to improve performances of neural agents playing a signaling game: a sender is exposed to an image and generates a sequence of symbols that is transmitted to a receiver, which uses it to distinguish between two images, one that is semantically related to the original image, and one that is not. We consider multiple design choices, such as pretraining the visual components of the agents, introducing regularization terms, how to sample training items from the dataset, and we study how these different choices impact the behavior and performances of the agents. To that end, we introduce a number of automated metrics to measure the properties of the emergent language. We find that some implementation choices are always beneficial, and that the information that is conveyed by the agents{'} messages is shaped not only by the game, but also by the overall design of the agents as well as seemingly unrelated implementation choices.",
}
```
