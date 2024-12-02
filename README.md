# AI-Based Attendance System

This project is a fully AI-based attendance system that uses facial recognition to mark attendance. It leverages OpenCV for video capture, DeepFace for facial recognition, and OpenAI's GPT-3 for data querying.

## Features

- Real-time facial recognition
- Attendance marking
- Data querying using OpenAI's GPT-3

## Requirements

- Python 3.6+
- OpenCV
- DeepFace
- Pandas
- OpenAI

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/ai-attendance-system.git
   cd ai-attendance-system
   ```

2. Create a virtual environment and activate it:
   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have a CSV file named `attendence.csv` in the project directory with the following format:
   ```csv
   Name,Roll No,Date1,Date2,...
   Alice,1
   Bob,2
   Charlie,3
   ...
   ```

2. Run the main script:
   ```sh
   python main.py
   ```

3. Follow the prompts to either recognize faces or get information from the CSV file.

## Contributing

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) for details on our code of conduct.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or concerns regarding this project, please contact me at aayush.ojha.dev@gmail.com .