# Maritime VTS - Intelligent Vessel Traffic Management System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![arXiv](https://img.shields.io/badge/arXiv-2505.00989-b31b1b.svg)](https://arxiv.org/abs/2505.00989)

ITSC2025 Submitted Paper #549 Official Demo Presentation

> An intelligent vessel traffic management system based on large language models, supporting natural language queries, collision analysis, traffic flow analysis, and more.

Cite us!!!
```bib
@article{sun2025vts,
  title={VTS-LLM: Domain-Adaptive LLM Agent for Enhancing Awareness in Vessel Traffic Services through Natural Language},
  author={Sun, Sijin and Zhao, Liangbin and Deng, Ming and Fu, Xiuju},
  journal={arXiv preprint arXiv:2505.00989},
  year={2025}
}
```

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Demo Video](#demo-video)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🚢 Project Overview

Maritime VTS is an intelligent vessel traffic management system based on large language models, designed to provide intelligent solutions for maritime supervision and vessel traffic services. The system integrates natural language processing, geographic information systems, real-time data analysis, and other technologies to support multiple query methods and analysis functions.

### Key Features

- **Natural Language Queries**: Support Chinese natural language queries for vessel information
- **Intelligent SQL Generation**: Automatically convert natural language to SQL queries
- **Real-time Data Analysis**: Provide traffic flow, collision risk, and other real-time analysis
- **Geographic Information Integration**: Integrate port, waterway, anchorage, and other geographic information data
- **Multi-model Support**: Support OpenAI, Ollama, OpenRoute, and other LLM models

## ✨ Features

### Core Functions
- 🔍 **Intelligent Queries**: Natural language to SQL queries
- 📊 **Data Analysis**: Traffic flow, collision analysis, data visualization
- 🗺️ **Geographic Information**: Port, waterway, anchorage information queries
- 🤖 **Entity Recognition**: Automatic recognition of vessels, ports, and other entities
- 🔧 **SQL Repair**: Automatic detection and repair of SQL syntax errors
- 💡 **Knowledge Reasoning**: Intelligent reasoning based on knowledge base

### Technology Stack
- **Backend**: Flask, PyMySQL, Pandas
- **AI Models**: OpenAI GPT, SQLCoder, Custom LLM
- **Database**: MySQL 8.0+
- **Geographic Information**: Shapefile (.shp) data
- **Knowledge Base**: RAGFlow

## 🎬 Demo Video

![Demo演示](./demo/demo.gif)

> Full demo video: [demo.mp4](./demo/demo.mp4)

## 🚀 Quick Start

### Requirements

- Python 3.8+
- MySQL 8.0+
- At least 4GB RAM
- Internet connection (for API calls)

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/maritime-vts.git
cd maritime-vts
```

2. **Install dependencies**
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install AI model dependencies
cd maritime-qa/scoder
pip install -r requirements.txt
```

3. **Configure database**
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE ship;
```

4. **Import geographic data**
```bash
cd llm-vts/data
python load_shp_data.py
```

5. **Configure API keys**
```bash
# Edit configuration file
cp llm-vts/config.yaml.example llm-vts/config.yaml
# Add your API keys in config.yaml
```

### Start Services

1. **Start LLM-VTS service**
```bash
cd llm-vts
python app.py
```

2. **Start RAGFlow service**

# Refer to RAGFlow official documentation

> https://github.com/infiniflow/ragflow


## 📁 Project Structure

```
maritime-vts/
├── demo/                    # Demo files
│   ├── demo.gif
│   └── demo.mp4
├── llm-vts/                # Main application module
│   ├── app.py              # Flask application main file
│   ├── config.yaml         # Configuration file
│   ├── data/               # Geographic data
│   │   ├── general/        # General geographic data
│   │   ├── port/           # Port data
│   │   ├── rule/           # Rule data
│   │   └── load_shp_data.py
│   └── util/               # Utility modules
│       ├── chat.py         # Chat functionality
│       ├── config.py       # Configuration management
│       ├── ner.py          # Entity recognition
│       └── web.py          # Web utilities
├── maritime-qa/            # Q&A system module
│   ├── ours.py             # Main model
│   ├── dsql_llm.py        # DSQL model
│   ├── sqlcoder_llm.py    # SQLCoder model
│   ├── api.yaml           # API configuration
│   ├── data/              # Dataset
│   ├── module/            # Function modules
│   ├── prompt/            # Prompts
│   ├── utils/             # Utility functions
│   └── scoder/            # SQLCoder integration
├── ragflow/               # RAGFlow configuration
└── README.md
```

## 📚 API Documentation

### Core API Endpoints

#### 1. Time Management
```http
POST /set_time
Content-Type: application/json

{
  "dt": "2025-01-01T12:00:00"
}
```

#### 2. Traffic Flow Analysis
```http
POST /traffic_flow
Content-Type: application/json

{
  "sql": "SELECT * FROM ship_ais WHERE timestamp > '2025-01-01'"
}
```

#### 3. Collision Analysis
```http
POST /collision_analysis
Content-Type: application/json

{
  "sql": "SELECT * FROM collision_events"
}
```

#### 4. Data Visualization
```http
POST /data_visualization
Content-Type: application/json

{
  "sql_statment": "SELECT * FROM ship_ais",
  "code": "result = df.groupby('vessel_type').count()"
}
```

#### 5. Entity Recognition
```http
POST /ner
Content-Type: application/json

{
  "entity": "Vessel A,Port B"
}
```

#### 6. SQL Query
```http
POST /sql
Content-Type: application/json

{
  "sql": "SELECT * FROM ship_ais WHERE mmsi = 123456789"
}
```

#### 7. SQL Validation
```http
POST /check_sql
Content-Type: application/json

{
  "sql": "SELECT * FROM ship_ais"
}
```

#### 8. SQL Repair
```http
POST /fix_sql
Content-Type: application/json

{
  "question": "Query the position of Vessel A",
  "sql": "SELECT * FROM ship_ais WHERE mmsi = 123456789",
  "error": "Syntax error",
  "fix_time": 5
}
```

## 📊 Dataset

Our dataset is available at [VTS-SQL](https://huggingface.co/datasets/PassbyGrocer/vts-sql).

### Data Processing Steps

1. **Navigate to maritime-qa directory**
```bash
cd maritime-qa
```

2. **Process Excel to JSON**
Select VTS style type: `command`, `operational`, `natural`
```bash
python table_to_json.py
```

3. **Process original questions to 5 representation types**
This will take several minutes
```bash
python data_process.py
python dail_sql.py
```

4. **Test models**
Run `ours.py`, `dsql_llm.py` and `sqlcoder_llm.py` to test the models

> **Note**: Add API_KEY to [api.yaml](./maritime-qa/api.yaml). `OpenAI`, `Ollama` and `OpenRoute` are supported. Examples are provided for each.

## ⚙️ Configuration

### Main Configuration Files

#### 1. llm-vts/config.yaml
```yaml
mysql:
  host: "localhost"
  user: "root"
  password: "123456"
  database: "ship"
  port: 3306

api:
  api_key: your_api_key
  address: localhost:1080
  rag:
    agent_id: your_rag_agent_id
  maritime:
    agent_id: your_maritime_agent_id

flask:
  host: localhost
  port: 5001
  debug: True
```

#### 2. maritime-qa/api.yaml
```yaml
OURS:
  api_key: your_openai_key
  agent_id: your_agent_id
  base_url: https://api.openai.com/v1

DSQL:
  api_key: your_dsql_key
  base_url: your_dsql_url

SQLCODER:
  api_key: your_sqlcoder_key
  base_url: your_sqlcoder_url
```

## 📁 Data Description

### 1. Shapefile Data

Use [load_shp_data.py](./llm-vts/data/load_shp_data.py) to upload shp files to MySQL(>=8.0).

- [general](./llm-vts/data/general/) - General geographic data
- [rule](./llm-vts/data/rule/) - Rule data
- [port](./llm-vts/data/port/) - Port data

### 2. AIS Data

Considering that AIS data is confidential, it is recommended to use our own dataset for replication work.

### 3. Knowledge Data

View [VTS-SQL](https://huggingface.co/datasets/PassbyGrocer/vts-sql) for each required knowledge, add each knowledge into RAGFLOW system.

## 🤝 Contributing

We welcome all forms of contributions!

### Ways to Contribute

1. **Report Bugs**: Report issues in GitHub Issues
2. **Feature Suggestions**: Propose new feature ideas
3. **Code Contributions**: Submit Pull Requests
4. **Documentation Improvements**: Help improve documentation
5. **Testing**: Help test and validate features

### Development Environment Setup

1. Fork the project
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Code Standards

- Follow Python PEP 8 code standards
- Add appropriate comments and docstrings
- Ensure all tests pass
- Update relevant documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact Us

- **Project Homepage**: [GitHub Repository](https://github.com/your-username/maritime-vts)
- **Dataset**: [VTS-SQL Dataset](https://huggingface.co/datasets/PassbyGrocer/vts-sql)
- **Paper**: ITSC2025 #549

## 🙏 Acknowledgments

Thanks to all developers and researchers who contributed to this project.

---

**Last Updated**: 2025-05-03  
**Version**: 1.0.0
