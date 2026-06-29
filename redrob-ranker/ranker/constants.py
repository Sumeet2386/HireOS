"""
Reference constants for the Redrob AI Candidate Ranking System.

All lookup tables, skill lists, company sets, city maps, and tech release dates
used across the ranking pipeline. Centralised here so every module pulls from
one source of truth.
"""

from __future__ import annotations

from datetime import date

# ---------------------------------------------------------------------------
# Reference date for time-delta calculations
# ---------------------------------------------------------------------------
REFERENCE_DATE = date(2026, 6, 12)

# ---------------------------------------------------------------------------
# JD-relevant core AI / ML skills
# ---------------------------------------------------------------------------
CORE_AI_SKILLS: set[str] = {
    # Embeddings & retrieval
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "word2vec", "fasttext", "glove", "doc2vec",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "chromadb", "annoy", "scann", "vector database", "vector db",
    "semantic search", "dense retrieval", "information retrieval",
    "retrieval augmented generation", "rag",
    # NLP core
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "text mining",
    "sentiment analysis", "topic modeling", "text generation",
    "transformers", "hugging face", "huggingface",
    "bert", "roberta", "gpt", "t5", "llama", "mistral",
    # Ranking & recommendations
    "learning to rank", "ltr", "lambdamart", "ndcg", "mrr", "map",
    "recommendation system", "recommender system", "collaborative filtering",
    "content-based filtering", "ranking", "re-ranking", "reranking",
    "search relevance", "search engine",
    # ML core
    "machine learning", "deep learning", "neural network",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "random forest", "svm", "logistic regression",
    "feature engineering", "model evaluation", "hyperparameter tuning",
    "cross-validation", "a/b testing", "ab testing",
    # LLM & GenAI
    "llm", "large language model", "fine-tuning", "finetuning",
    "prompt engineering", "langchain", "llamaindex", "llama index",
    "openai", "chatgpt", "gpt-4", "gpt-3", "claude",
    "lora", "qlora", "peft", "rlhf", "dpo",
    "onnx", "model quantization", "model compression",
    "knowledge distillation",
    # Computer vision
    "computer vision", "image classification", "object detection",
    "image segmentation", "opencv",
}

# General engineering skills — not AI-specific, shouldn't inflate core AI count
GENERAL_ENGINEERING_SKILLS: set[str] = {
    "python", "pandas", "numpy", "scipy",
    "mlflow", "wandb", "weights and biases",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "airflow", "kubeflow", "mlops",
    "spark", "pyspark", "ray",
    "sql", "postgresql", "mongodb", "elasticsearch",
    "data pipeline", "etl", "data engineering",
}

# Subset of skills that are highly specific to the JD's core needs
HIGH_SIGNAL_SKILLS: set[str] = {
    "sentence-transformers", "sentence transformers", "bge", "e5",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "semantic search", "dense retrieval", "information retrieval",
    "rag", "retrieval augmented generation",
    "learning to rank", "ltr", "lambdamart",
    "recommendation system", "recommender system",
    "ranking", "re-ranking", "reranking", "search relevance",
    "nlp", "natural language processing",
    "transformers", "huggingface", "hugging face",
    "bert", "llm", "fine-tuning", "finetuning",
    "pytorch", "tensorflow",
    "ndcg", "mrr", "a/b testing",
}

# ---------------------------------------------------------------------------
# Skill synonyms for entailment checking
# ---------------------------------------------------------------------------
SKILL_SYNONYMS: dict[str, list[str]] = {
    "nlp": ["natural language processing", "text mining", "text classification", "ner", "sentiment"],
    "ml": ["machine learning", "deep learning", "neural network", "model training"],
    "pytorch": ["torch", "pytorch"],
    "tensorflow": ["tf", "tensorflow", "keras"],
    "bert": ["bert", "roberta", "distilbert", "transformers"],
    "llm": ["large language model", "gpt", "llama", "chatgpt", "openai", "fine-tun"],
    "rag": ["retrieval augmented", "retrieval-augmented", "rag pipeline"],
    "vector database": ["vector db", "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma"],
    "recommendation": ["recommender", "collaborative filtering", "content-based filtering"],
    "ranking": ["learning to rank", "ltr", "lambdamart", "re-rank", "rerank"],
    "embeddings": ["embedding", "sentence-transformer", "bge", "e5", "word2vec", "dense retrieval"],
    "docker": ["container", "docker", "kubernetes", "k8s"],
    "aws": ["amazon web services", "s3", "sagemaker", "ec2", "lambda"],
    "gcp": ["google cloud", "bigquery", "vertex ai"],
    "azure": ["microsoft azure", "azure ml"],
    "python": ["python", "django", "flask", "fastapi"],
    "sql": ["sql", "postgresql", "mysql", "snowflake", "bigquery", "data warehouse"],
    "spark": ["spark", "pyspark", "databricks"],
    "airflow": ["airflow", "orchestrat", "dag"],
    "react": ["react", "reactjs", "react.js"],
    "feature engineering": ["feature engineer", "feature extraction", "feature selection"],
    "evaluation": ["ndcg", "mrr", "map", "precision", "recall", "f1", "auc", "roc"],
}

# ---------------------------------------------------------------------------
# Consulting / IT services firms (for career background detection)
# ---------------------------------------------------------------------------
CONSULTING_FIRMS: set[str] = {
    "tcs", "tata consultancy services", "tata consultancy",
    "infosys", "infosys limited", "infosys bpo",
    "wipro", "wipro limited", "wipro technologies",
    "accenture", "accenture solutions",
    "cognizant", "cognizant technology solutions", "cts",
    "capgemini", "capgemini technology services",
    "hcl", "hcl technologies", "hcltech",
    "tech mahindra", "techmahindra",
    "mindtree", "ltimindtree", "lti mindtree",
    "mphasis",
    "l&t infotech", "lt infotech", "lti", "larsen & toubro infotech",
    "ibm", "ibm consulting", "ibm india",
    "deloitte", "deloitte consulting",
    "kpmg",
    "ey", "ernst & young", "ernst and young",
    "pwc", "pricewaterhousecoopers",
    "dxc technology", "dxc",
    "ntt data", "ntt",
    "hexaware", "hexaware technologies",
    "zensar", "zensar technologies",
    "persistent systems", "persistent",
    "cyient",
    "birlasoft",
    "sonata software",
    "niit technologies", "niit",
    "sasken",
    "mphasis",
    "atos",
}

# ---------------------------------------------------------------------------
# Known product companies (positive signal)
# From competitor repo + our additions
# ---------------------------------------------------------------------------
PRODUCT_BRANDS: set[str] = {
    # Big tech
    "google", "alphabet", "deepmind",
    "meta", "facebook", "instagram", "whatsapp",
    "amazon", "aws", "amazon web services",
    "microsoft", "linkedin", "github",
    "apple",
    "netflix",
    "uber",
    "airbnb",
    "spotify",
    "twitter", "x corp",
    "salesforce",
    "oracle",
    "adobe",
    "atlassian",
    "databricks",
    "snowflake",
    "stripe",
    "openai",
    # Indian product companies
    "flipkart",
    "swiggy",
    "zomato",
    "ola", "ola cabs",
    "paytm",
    "freshworks", "freshdesk",
    "razorpay",
    "meesho",
    "phonepe",
    "cred",
    "dream11",
    "unacademy",
    "upgrad",
    "byju's", "byjus",
    "zerodha",
    "groww",
    "sharechat",
    "nykaa",
    "urban company", "urbanclap",
    "myntra",
    "bigbasket",
    "dunzo",
    "lenskart",
    "policybazaar",
    "cars24",
    "oyo", "oyo rooms",
    "jio", "reliance jio",
    "makemytrip",
    "goibibo",
    "cleartrip",
    "practo",
    "1mg",
    "healthkart",
    # Global product companies
    "shopify",
    "twilio",
    "cloudflare",
    "palantir",
    "doordash",
    "lyft",
    "pinterest",
    "snap", "snapchat",
    "reddit",
    "discord",
    "notion",
    "figma",
    "canva",
    "zoom",
    "slack",
    "dropbox",
    "square", "block",
    "robinhood",
    "coinbase",
    "datadog",
    "elastic", "elasticsearch",
    "mongodb",
    "confluent",
    "hashicorp",
    "vercel",
    "supabase",
    "hugging face", "huggingface",
    "anthropic",
    "cohere",
    "stability ai",
}

# ---------------------------------------------------------------------------
# Product industry keywords (from competitor repo)
# ---------------------------------------------------------------------------
PRODUCT_INDUSTRY_KEYWORDS: tuple[str, ...] = (
    "software", "saas", "e-commerce", "ecommerce",
    "fintech", "marketplace", "recruit", "hr tech", "hr-tech",
    "adtech", "edtech", "consumer", "food delivery",
    "transportation", "insurance tech", "insurtech",
    "gaming", "ai/ml", "artificial intelligence",
    "conversational ai", "healthtech", "health tech",
    "proptech", "legaltech", "agritech",
    "cybersecurity", "cloud computing",
    "social media", "media streaming",
    "internet", "mobile", "platform",
)

# ---------------------------------------------------------------------------
# Tier 1 Indian cities
# ---------------------------------------------------------------------------
TIER1_INDIA_CITIES: set[str] = {
    "pune", "noida", "greater noida",
    "hyderabad", "secunderabad",
    "mumbai", "navi mumbai", "thane",
    "delhi", "new delhi",
    "gurgaon", "gurugram",
    "bangalore", "bengaluru",
    "chennai",
    "kolkata",
    "ahmedabad",
    "jaipur",
    "chandigarh",
    "kochi", "cochin",
    "thiruvananthapuram", "trivandrum",
    "indore",
    "lucknow",
    "coimbatore",
    "visakhapatnam", "vizag",
    "nagpur",
    "bhopal",
    "goa", "panaji",
}

# ---------------------------------------------------------------------------
# Title classification
# ---------------------------------------------------------------------------
HIGH_SIGNAL_TITLES: tuple[str, ...] = (
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer",
    "senior machine learning engineer",
    "nlp engineer", "search engineer", "relevance engineer",
    "ranking engineer", "recommendation engineer",
    "data scientist", "senior data scientist",
    "applied scientist", "research engineer",
    "research scientist", "deep learning engineer",
    "computer vision engineer", "speech engineer",
    # Added: titles that were falling through to 0.25 default
    "ai specialist", "ai research engineer", "applied ml engineer",
    "senior nlp engineer", "lead ai engineer",
    "staff machine learning engineer", "staff ml engineer",
    "recommendation systems engineer", "junior ml engineer",
)

ADJACENT_SIGNAL_TITLES: tuple[str, ...] = (
    "software engineer", "senior software engineer",
    "staff software engineer", "principal engineer",
    "backend engineer", "senior backend engineer",
    "data engineer", "senior data engineer",
    "analytics engineer", "platform engineer",
    "full stack developer", "full stack engineer",
    "python developer", "java developer",
    "cloud engineer", "devops engineer",
    "site reliability engineer", "sre",
)

NON_TECHNICAL_TITLES: tuple[str, ...] = (
    "marketing manager", "hr manager", "human resources",
    "accountant", "content writer", "copywriter",
    "sales executive", "sales manager", "sales representative",
    "graphic designer", "ui designer",
    "mechanical engineer", "civil engineer",
    "electrical engineer", "chemical engineer",
    "customer support", "customer service",
    "operations manager", "operations executive",
    "project manager", "program manager",
    "business analyst", "business development",
    "financial analyst", "finance manager",
    "administrative assistant", "executive assistant",
    "teacher", "professor", "lecturer",
    "lawyer", "legal counsel",
    "doctor", "physician", "nurse",
    "architect",  # building architect, not software
    "pharmacist", "dentist",
    "journalist", "reporter",
    "chef", "cook",
    "driver", "delivery",
)

# ---------------------------------------------------------------------------
# Tech release years (for skill maturity validation / honeypot detection)
# ---------------------------------------------------------------------------
TECH_RELEASE_YEARS: dict[str, int] = {
    # LLM era
    "langchain": 2022,
    "llamaindex": 2022,
    "llama index": 2022,
    "chatgpt": 2022,
    "gpt-4": 2023,
    "gpt-3.5": 2022,
    "gpt-3": 2020,
    "claude": 2023,
    "gemini": 2023,
    "llama": 2023,
    "llama 2": 2023,
    "llama 3": 2024,
    "mistral": 2023,
    "mixtral": 2023,
    "phi": 2023,
    "qwen": 2023,
    "lora": 2021,
    "qlora": 2023,
    "peft": 2022,
    "rlhf": 2022,
    "dpo": 2023,
    "rag": 2020,
    "retrieval augmented generation": 2020,
    # Embedding models
    "bge": 2023,
    "e5": 2022,
    "sentence-transformers": 2019,
    "sentence transformers": 2019,
    # Vector DBs
    "pinecone": 2021,
    "weaviate": 2021,
    "qdrant": 2021,
    "milvus": 2019,
    "chroma": 2022,
    "chromadb": 2022,
    # Older tech (generous dates)
    "bert": 2018,
    "transformers": 2018,
    "pytorch": 2016,
    "tensorflow": 2015,
    "docker": 2013,
    "kubernetes": 2014,
    "spark": 2014,
    "airflow": 2015,
    "fastapi": 2018,
    "react": 2013,
    "node.js": 2009,
    "python": 1991,
}

# ---------------------------------------------------------------------------
# Negative title patterns (non-engineering roles)
# ---------------------------------------------------------------------------
NEGATIVE_TITLE_PATTERNS: tuple[str, ...] = (
    "marketing", "hr ", "human resource",
    "accountant", "accounting", "finance manager",
    "content writer", "copywriter",
    "sales", "business development",
    "graphic designer", "ui/ux",
    "mechanical engineer", "civil engineer",
    "electrical engineer", "chemical engineer",
    "customer support", "customer service",
    "operations manager",
    "teacher", "professor", "lecturer",
    "lawyer", "legal",
    "doctor", "physician", "nurse",
    "chef", "cook", "driver",
)
