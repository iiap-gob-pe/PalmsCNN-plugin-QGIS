# 🌴 PalmsCNN: Palm Tree Detection (QGIS Plugin)

[![QGIS](https://img.shields.io/badge/QGIS-3.34%2B-green.svg)](https://qgis.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-success.svg)]()

**PalmsCNN** is a QGIS plugin that applies **Deep Learning models** to automatically detect three ecologically and economically important Amazonian palm species:  
**aguaje** (*Mauritia flexuosa*), **huasai** (*Euterpe precatoria*), and **ungurahui** (*Oenocarpus bataua*)  
(Tagle et al., 2025, [*Nature Communications*](https://www.nature.com/articles/s41467-025-58358-5)).

This plugin was developed within the framework of the projects *“Supervisiones Optimizadas”* and *“New approaches to understand the state of biodiversity and contribute to social well-being: studying the distribution and degradation of Mauritia flexuosa in the Amazon”*, through the collaboration of **OSINFOR**, **IIAP**, **SERNANP**, the **University of Leeds**, the **University of Brescia**, and **Wageningen University**, with financial support from **Newton Fund**, **FONDECYT**, **WWF**, **GIZ**, and **USAID**.

---

## 🧠 Overview

PalmsCNN uses **Convolutional Neural Networks (CNNs)** exported to the **ONNX** format to recognize individual palm crowns in RGB orthomosaic images captured by drones or high-resolution satellites.  
The plugin enables automated, reproducible, and cost-efficient mapping of Amazonian palm ecosystems.

---

## 🌿 Target Species

| Scientific name | Common name |
|------------------|-------------|
| *Mauritia flexuosa* | Aguaje |
| *Euterpe precatoria* | Huasai |
| *Oenocarpus bataua* | Ungurahui |

---

## ⚙️ Key Features

- Automatic palm detection from **RGB imagery** (no multispectral data required).  
- Pre-trained CNN models in **ONNX** format.  
- Full integration with the **QGIS Processing Toolbox**.  
- Georeferenced output layers (vector or raster).  
- Cross-platform support (**Windows, Linux, macOS**).  
- Automatic setup of a Python **virtual environment (venv)** for dependencies.  
---

## 📦 Installation

### 🔹 From ZIP (for end users)
1. Download the latest release from [Releases](../../releases).  
2. In QGIS, open:  
   `Plugins → Manage and Install Plugins → Install from ZIP`.  
3. Select the file:  
   `deteccion_de_palmeras-<version>.zip`  
4. Click **Install Plugin**.

### 🔹 From source (for developers)
```bash
git clone https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS.git
d PalmsCNN-plugin-QGIS/deteccion_de_palmeras/help
make package        # or make.bat package on Windows
```
---
## 🌍 Credits

**Instituto de Investigaciones de la Amazonía Peruana (IIAP)**
*Laboratorio de Inteligencia Artificial* - Programa BOSQUES
Iquitos, Peru

Developed by the technical team of the **IIAP**, in collaboration with **OSINFOR**, **SERNANP**, **University of Leeds**, **University of Brescia**, and **Wageningen University**, under the projects *“Supervisiones Optimizadas”* and *“New approaches to understand the state of biodiversity and contribute to social well-being.”*

Funding provided by **Newton Fund**, **FONDECYT**, **WWF**, **GIZ**, and **USAID**.

---

## 🧾 License

This project is distributed under the [MIT License](./LICENSE).

> © 2025 Instituto de Investigaciones de la Amazonía Peruana (IIAP).
> Free for scientific, educational, and conservation use.

---

## 🔗 References

* Tagle, X., Cárdenas, R., Palacios, S., Sanjurjo, J., et al. (2025). *Deep learning enables scalable mapping of Mauritia flexuosa degradation across the Amazon Basin.* **Nature Communications**, 14, 58358. [https://doi.org/10.1038/s41467-025-58358-5](https://doi.org/10.1038/s41467-025-58358-5)
* QGIS Documentation — [https://docs.qgis.org](https://docs.qgis.org)
* ONNX Runtime — [https://onnxruntime.ai](https://onnxruntime.ai)
* IIAP — [https://www.iiap.gob.pe](https://www.iiap.gob.pe)

---

## 📚 How to Cite

If you use **PalmsCNN** in a scientific publication, please cite:

> Palacios, S., Cárdenas, R., Torres, S., & Tagle, X. (2025).
> *PalmsCNN: Palm Tree Detection (QGIS Plugin).*
> Instituto de Investigaciones de la Amazonía Peruana (IIAP).
> Available at: [https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS](https://github.com/iiap-gob-pe/PalmsCNN-plugin-QGIS)
> Contact: [rcardenasv@iiap.gob.pe](mailto:rcardenasv@iiap.gob.pe)
>
>Tagle Casapia, X., Cardenas-Vigo, R., Marcos, D. et al. *Effective integration of drone technology for mapping and managing palm species in the Peruvian Amazon*. **Nat Commun 16**, 3764 (2025). [https://doi.org/10.1038/s41467-025-58358-5](https://doi.org/10.1038/s41467-025-58358-5)
> 