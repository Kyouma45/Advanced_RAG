# Advanced_RAG

**Advanced_RAG** is a command-line application leveraging Retrieval-Augmented Generation (RAG) using Large Language Models (LLMs) and LangChain. This application allows users to store and retrieve projects without the need for PDFs, provides enhanced visual interfaces, and features user-specific memory for human-like interactions.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Screenshots](#screenshots)
7. [Contributing](#contributing)
8. [License](#license)

## Introduction

Welcome to **Advanced_RAG**! This project aims to enhance productivity by integrating Retrieval-Augmented Generation (RAG) technology. It uses LLMs to enable advanced project storage, retrieval, and user interaction capabilities.

## Features

- Retrieval-Augmented Generation using LLMs and LangChain.
- Project storage and retrieval without PDFs.
- Enhanced visual interfaces.
- User-specific memory for contextual and follow-up questions.
- Support for both GPT-4o and Gemini-1.5pro (open-source).

## Installation

### Prerequisites

Ensure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

### Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Advanced_RAG.git
   cd Advanced_RAG
2. Install the dependencies:
   ```sh
   pip install -r requirements.txt
3. Create a `.env` file in the root directory with the following keys:
   ```sh
    GOOGLE_API_KEY=""
    LANGCHAIN_API_KEY=""
    OPENAI_API_KEY=""
4. To switch to the free version using Gemini-1.5pro, uncomment lines 82 and 83 and comment out lines 84 and 85 in the code.

## Usage
Running the Application
1. Open your terminal and navigate to the project directory.
2. Run the following command:
   ```sh
   typer v3.py run
3. Follow the on-screen instructions to interact with the application.

## Configuration
Adjust the configuration settings in your `.env` file to personalize your API keys and choose your preferred LLM.

## Screenshots
Here's a glimpse of how the application looks in action:
![Screenshot from 2024-08-02 15-48-01](https://github.com/user-attachments/assets/985e679a-13e0-4eef-98e6-0d34a46ceef0)
![Screenshot from 2024-08-02 15-49-48](https://github.com/user-attachments/assets/a34baa31-17d1-4c6b-8b7f-19d294a3c267)
As you can see, it can answer questions along with follow-up questions using its memory capabilities. It also stores projects so you can load a project later and ask questions even without the PDFs.

## Contributing
We welcome contributions from the community! Please read the Contributing Guidelines for more information on how to get involved.

## License
This project is licensed under the Apache License.

