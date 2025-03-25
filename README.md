# CourtKeyNet

## A Novel Octave-Based Architecture for Badminton Court Detection with Geometric Constraints

![CourtKeyNet Architecture](https://github.com/adithyanraj03/courtkeynet/raw/main/images/architecture.png)

## Research Status

**This is an ongoing research project.** The repository will be progressively updated with implementation code, pre-trained models, and comprehensive documentation as the research advances toward publication. Thank you for your interest in this work.

## Abstract

Court detection plays a crucial role in sports video analysis, match statistics generation, and automated broadcasting systems. Existing approaches often rely on generic object detection methods, which lack the domain-specific understanding required for precise court boundary localization. This research introduces CourtKeyNet, a novel deep learning architecture specifically designed for badminton court detection. 

Unlike existing models that rely on general-purpose object detection frameworks, CourtKeyNet incorporates several innovative components: (1) an Octave Feature Extractor that processes visual information at multiple frequency bands; (2) a Polar Transform Attention mechanism that enhances boundary detection; (3) a Quadrilateral Constraint Module that ensures geometric consistency; and (4) a novel Geometric Consistency Loss function. 

Experimental validation demonstrates that CourtKeyNet significantly outperforms general-purpose keypoint detection approaches in mean Keypoint Localization Accuracy and in Court Detection IoU. Furthermore, our architecture is specifically designed with an open-source license suitable for commercial applications, unlike existing GPL-licensed alternatives.

## Key Innovations

- **Octave Feature Extractor**: Processes visual information at multiple frequency bands simultaneously to capture both fine details and global structure
- **Polar Transform Attention**: Enhances boundary detection by transforming features to polar space
- **Quadrilateral Constraint Module**: Enforces geometric consistency among detected keypoints
- **Geometric Consistency Loss**: Promotes proper quadrilateral properties during training

## Repository Structure

The repository will include:

- PyTorch implementation of the CourtKeyNet architecture
- Training and evaluation scripts
- Pre-trained models for various scenarios
- Custom-built annotation tool for court keypoint labeling
- Comprehensive dataset with precise keypoint annotations
- Evaluation metrics and comparison with state-of-the-art methods

## License

This project is released under [MIT License](LICENSE), making it suitable for both academic and commercial applications.

## Citation

If you find this work useful for your research, please consider citing:

```
@article{raj2025courtkeynet,
  title={CourtKeyNet: A Novel Octave-Based Architecture for Badminton Court Detection with Geometric Constraints},
  author={Raj, Adithya N},
  journal={arXiv preprint},
  year={2025}
}
```

## Contact

For questions or collaboration opportunities, please contact Adithya N Raj at adithyanraj03@gmail.com

---

**Note**: Full implementation details, experimental results, and comprehensive documentation will be available soon. The current repository represents preliminary work toward a complete open-source solution for badminton court detection.
