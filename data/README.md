# 🚀 algo-engineering-cpp

[![CI Pipeline](https://github.com/Aditya-kumar-yadav/algo-engineering-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/Aditya-kumar-yadav/algo-engineering-cpp/actions)
[![Language](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![Build System](https://img.shields.io/badge/Build-CMake-orange.svg)](https://cmake.org/)
[![Testing](https://img.shields.io/badge/Testing-GoogleTest-green.svg)](https://github.com/google/googletest)

A scalable, fully automated architecture for solving algorithmic problems using modern software engineering practices.

This repository transcends standard competitive programming storage by enforcing production-level constraints, including **Test-Driven Development (TDD)**, **Continuous Integration (CI)**, and **Dynamic Build Systems**. 

The goal is to treat algorithmic problem-solving with the exact engineering discipline required in top-tier corporate environments.

---

## ✨ Core Features

* **Modern Standards:** Highly optimized **C++17** solutions.
* **Automated Validation:** Strict unit testing via **Google Test**.
* **Dynamic Compilation:** Zero-touch build system via **CMake**.
* **CI/CD Integration:** Autonomous pipeline execution via **GitHub Actions**.
* **Self-Documenting:** Python-driven metadata extraction and markdown generation.

---

## 🏗️ System Architecture

```text
📦 algo-engineering-cpp
 ┣ 📂 .github/workflows   # GitHub Actions CI/CD pipelines
 ┣ 📂 hubs                # Auto-generated company & topic indexes
 ┣ 📂 metadata            # Core JSON database tracking all solutions
 ┣ 📂 scripts             # Python automation and pipeline utilities
 ┣ 📂 src                 # C++ algorithmic implementations (.hpp)
 ┣ 📂 tests               # Google Test assertions (_test.cpp)
 ┣ 📜 CMakeLists.txt      # Dynamic build configuration
 ┣ 📜 CONTRIBUTING.md     # Developer workflow and Git hygiene rules
 ┗ 📜 README.md           # Autonomously updated entry point
```

---

## ⚙️ The Engineering Lifecycle

```text
[ Problem Solved ] ➔ [ Google Tests Written ] ➔ [ Git Commit ] ➔ [ GitHub Push ]
                                                                        ⬇
[ Docs Regenerated ] ⬅ [ Metadata Extracted ] ⬅ [ CMake Build + Tests Pass ]
```

### 🔧 Automation Tooling (Local & CI)
The `/scripts` directory operates as the backbone of this repository:
* `start.py`: Scaffolds C++ boilerplate, test files, and ensures standardized naming.
* `auto_discover.py`: Acts as a gatekeeper, scanning for duplicate IDs and validating JSON metadata.
* `generate_docs.py`: Autonomously builds Markdown tables, hubs, and repository indexes.

---

## 🛠️ Local Development Setup

### Prerequisites
* **CMake** (v3.14+)
* **Python** (v3.10+)
* **C++17** compatible compiler (GCC, Clang, or MSVC)

### Quick Start
```bash
# 1. Clone the repository
git clone [https://github.com/Aditya-kumar-yadav/algo-engineering-cpp.git](https://github.com/Aditya-kumar-yadav/algo-engineering-cpp.git)
cd algo-engineering-cpp

# 2. Configure the build environment
cmake -B build -S .

# 3. Compile the project
cmake --build build

# 4. Execute the test suite
cd build && ctest --output-on-failure
```

---

## 📊 Problem Database & Analytics

Our CI/CD pipeline autonomously maintains a central index tracking difficulty curves, company-wise frequencies, and algorithm categorization. 

📄 **[View Complete Solution Index](./src/README.md)** *(Auto-generated upon merge)*

---

## 🤝 Contributing

We enforce strict rules regarding git hygiene, build modification, and testing to keep the main branch pristine. 

If you are interested in optimizing existing logic or expanding the test coverage, please read our **[Contributing Guidelines](CONTRIBUTING.md)** before opening a Pull Request.

---
## 📫 Connect

Built for developers interested in backend systems, C++ architecture, and automated workflows. Feel free to explore the codebase or reach out to discuss scalable engineering.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/aditya-yadav-43442b323/)
[![LeetCode](https://img.shields.io/badge/LeetCode-Profile-orange?logo=leetcode)](https://leetcode.com/u/AdiTheOP/)