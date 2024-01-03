# this-is-not-real-aerial-imagery

## Table of Contents

1. [Context](#overview)
   - [End of Year Project](#end-of-year-project)
     - [Constraints](#contraints)
     - [Choice](#choice)
   - [Necessity to build Full stack app](#necessity-to-build-full-stack-app)
   
2. [Installation](#installation)

3. [Usage](#usage)
4. [Output and model description](#output-and-model-description)
5. [Acknowledgements](#acknowledgements)
6. [License](#license)
7. [Contributing](#contributing)

## Context

### End Of Year Project

As part of my final year as a Machine Learning Scientist, I embarked on the journey of creating a comprehensive project that encapsulates all the knowledge I've gained throughout the year. Having developed a keen interest in the field of computer vision and deep learning, I believe it is a fundamental cornerstone for building future applications in computer vision and augmented reality. It was only natural for me to choose a project within this domain to further enhance my understanding. Leveraging my knowledge from the last project (ViT), I decided to build a Diffusion model from scratch.

The entire project, from conception to coding, debugging, and deployment, unfolded within a tight 4-week timeline, constrained by impending vacations.

#### Constraints

Before delving into the project, I imposed a set of constraints to add an extra layer of challenge:

- **No Public Datasets**: I committed to gathering all the necessary data myself.
- **No School-Learned Deep Learning Frameworks**: I opted not to use any deep learning frameworks taught in school.
- **No Third-Party Software for Data Gathering**: I refrained from utilizing any third-party software for data collection.

#### Choice

I chose this project because of my fascination with satellite imagery of Earth. Having previously worked on classification and segmentation projects with satellite imagery, I decided to venture into the realm of diffusion for this new adventure.

## Installation

To use this codebase, follow these steps:
```bash
git clone https://github.com/Camaltra/this-is-not-real-aerial-imagery.git
cd this-is-not-real-aerial-imagery
```

Please refer to the README in the src/ folder for additional installation steps and environment setup required for different modules.

## Usage

Please refer to sub folder README in the `src/` folder to see all usage for the differents modules.   
Short summary of the modules:  

- ETL: Gather data from Google Earth web application.

    - ETL/MODEL: Model Registry for experiments classification models
- SERVER: Back-end Server to serve the Front End Application
- AI: Model and training for the Diffusion Model.

## Output and model description:
For details on the model architecture and output, please refer to the linked blog post that provides comprehensive information.

## Acknowledgements

This project draws inspiration from the following works:

- [DDPM Paper](https://arxiv.org/abs/2006.11239)
- [Group Norm Paper](https://arxiv.org/abs/1803.08494)
- [ConvNext Paper](https://arxiv.org/abs/2201.03545)
- [UNet Paper](https://arxiv.org/abs/1505.04597)
- [ACC Unet Paper](https://arxiv.org/abs/2308.13680)
- [LeNet Paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
- [AlexNet Paper](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj92J_hjsKDAxVyT6QEHTKwA5gQFnoECA4QAQ&url=https%3A%2F%2Fproceedings.neurips.cc%2Fpaper%2F4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf&usg=AOvVaw26V5YkBm0FS972qI4eBNgu&opi=89978449)
- [VGG Paper](https://arxiv.org/abs/1409.1556)  

We acknowledge their significant contributions to the field.

## License

This project is licensed under the [Apache 2.0] - see the LICENSE.md file for details.

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. We welcome any feedback, suggestions, or improvements.

Happy deep fake generation with the Diffusion Model!

