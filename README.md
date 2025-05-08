# LonelyLittleMeowSounds 😿🎶
A Dockerized Python purr-ocessor that leverages LangChain to process text from files (PDFs, CSVs, DOCX, XLSX) and takes large language models (LLMs) to the next level. Built with Python 3.13 and docker-compose, this project is your friendly companion for making LLMs useful in daily life through efficient text handling and beyond.

## 🌟 Features
- Text Processing: Splits text from multiple file formats (PDF, CSV, DOCX, XLSX) into chunks using LangChain’s text splitters.
- Dockerized Setup: Runs seamlessly with a Dockerfile and docker-compose.yml for easy deployment and scalability.
- Python 3.13: Harnesses the latest Python features for modern, efficient code.
- LLM-Ready: Designed to power LLM-driven tasks, making daily workflows smarter and more productive.
- Extensible: Built to grow with future LLM features like embeddings, chat, or retrieval-augmented generation (RAG).

## 🚀 Getting Started

#### Prerequisites
- Docker and Docker Compose
- Python 3.13 (if running locally without Docker)
#### Installation
- Clone the repo:
```bash
git clone https://github.com/KHARAPSY/LonelyLittleMeowSounds
cd LonelyLittleMeowSounds
```
- Build and run with Docker Compose:
```bash
docker-compose up --build
```
- (Optional) Run locally:
```bash
pip install -r requirements.txt
python main.py
```
#### Usage
- Drop your files (PDF, CSV, DOCX, XLSX) into the input directory.
- Run the service to process files and generate text chunks for LLM applications.
- Check the output directory for results or integrate with your favorite LLM pipeline.

See `docs/` for detailed configuration and examples.

## 🛠️ Tech Stack
- Python 3.13: Cutting-edge Python for robust performance.
- LangChain: Text splitting and processing for LLM workflows.
- Docker: Containerized setup for consistency and portability.
- docker-compose: Simplified multi-container management.

## 📝 License
This project is licensed under the MIT License [LICENSE](LICENSE) for open collaboration. 
<!-- It uses [LangChain](https://github.com/langchain-ai/langchain), licensed under the MIT License. -->

## 🤝 Contributing
Meow! Contributions are welcome. To get started:
- Fork the repo.
- Create a feature branch ```git checkout -b feature/AmazingMeow```.
- Commit your changes ```git commit -m 'Add some meowtastic feature'```.
- Push to the branch ```git push origin feature/AmazingMeow```.
- Open a pull request.
Please follow the Code of Conduct [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and check the contributing guidelines [CONTRIBUTING.md](CONTRIBUTING.md).

## 🌈 Why LonelyLittleMeowSounds?
This project is a purr-sonal exploration of making LLMs practical and fun for everyday tasks. Whether you’re chunking documents or dreaming of AI-powered workflows, LonelyLittleMeowSounds is here to make your code (and life) a little less lonely.

## 📬 Contact
Got questions or ideas? Reach out via GitHub Issues or connect with me at [KHARAPSY](https://x.com/KHARAPSY).
