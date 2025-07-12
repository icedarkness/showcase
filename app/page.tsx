import Head from 'next/head';

export default function HomePage() {
  const projects = [
    {
      title: "Lending Club Credit Risk Model with XAI",
      description: "This project demonstrates a complete, end-to-end workflow for building and interpreting a credit risk model on the Lending Club dataset. The primary objective is to predict loan defaults (\"Charged Off\") while maintaining a strong focus on Explainable AI (XAI). The analysis begins with memory-efficient data ingestion and preprocessing of a large dataset, followed by feature engineering to create impactful predictors. A LightGBM model is trained and evaluated using metrics suitable for imbalanced data, such as ROC AUC and Precision-Recall curves. The core of the project lies in using the SHAP library to interpret the model's decisions, providing both global feature importance and local, instance-level explanations for why a specific loan is predicted to default or be paid off.",
      technologies: "Python, Pandas, LightGBM, SHAP, Scikit-learn, Jupyter Notebook",
      githubLink: "https://github.com/icedarkness/showcase/blob/main/Lending%20Club%20Credit%20Risk%20Model%20with%20XAI/Lending%20Club%20Credit%20Risk%20Model%20with%20XAI.ipynb"
    },
    {
      title: "RAG-Powered Chatbot for Financial Product Information",
      description: "This project features a web-based chatbot designed to answer user questions about a fictional bank's financial products. It leverages a Retrieval-Augmented Generation (RAG) pipeline to ensure answers are accurate and grounded in a specific knowledge base. The application is built as a self-contained HTML file for easy deployment and showcases modern AI techniques, including using `sentence-transformers` for embeddings and `FAISS` for efficient vector search. A key feature is the chatbot's ability to cite its sources, providing transparency by showing the user which part of the knowledge base was used to generate the answer.",
      technologies: "RAG, Gemini API, Sentence-Transformers, FAISS, HTML, CSS",
      githubLink: "https://github.com/icedarkness/showcase/tree/main/RAG-Powered%20Financial%20Chatbot"
    },
    {
      title: "German Credit Risk Model with XAI",
      description: "This project involves building a highly accurate classification model to predict credit risk using the well-known German Credit Data (Statlog) dataset. The workflow begins with a thorough exploratory data analysis (EDA) to understand the relationships between features like age, credit purpose, and loan duration. A robust preprocessing pipeline is created using scikit-learn's `ColumnTransformer` to handle numerical and categorical data appropriately. A RandomForestClassifier is trained and evaluated, but the core focus is on Explainable AI (XAI). The SHAP library is used to interpret the model's decisions, providing both global feature importance and local, instance-level explanations to answer questions like, \"Why was this specific applicant flagged as a high risk?\"",
      technologies: "Python, Pandas, Scikit-learn, RandomForest, SHAP, Matplotlib, Seaborn",
      githubLink: "https://github.com/icedarkness/showcase/blob/main/German%20Credit%20Data%20(Statlog)/German_Credit_Data.ipynb"
    },
    {
      title: "Customer Segmentation for Targeted Marketing in Lending",
      description: "This project showcases the use of unsupervised machine learning to drive business strategy. Using the classic \"Mall Customer\" dataset, this analysis applies K-Means clustering to segment customers based on their financial profiles, specifically annual income and spending score. The workflow includes a thorough exploratory data analysis (EDA), using the Elbow Method to determine the optimal number of clusters, and building the final segmentation model. The core of the project is the interpretation of these clusters, where distinct customer personas (e.g., \"The Savers,\" \"The High Rollers\") are created. Finally, actionable, targeted marketing strategies are proposed for each segment from a lending perspective, demonstrating how data-driven insights can inform business decisions.",
      technologies: "Python, Pandas, Scikit-learn (K-Means, StandardScaler), Matplotlib, Seaborn",
      githubLink: "https://github.com/icedarkness/showcase/blob/main/Customer%20Segmentation%20for%20Marketing/Customer%20Segmentation%20for%20Marketing.ipynb"
    },
    {
      title: "Time-Series Forecasting of Loan Portfolio Performance",
      description: "This project showcases time-series analysis and forecasting, a critical skill for financial planning and risk management. A realistic, synthetic dataset of monthly loan defaults is generated to include a clear trend and seasonality, mimicking real-world portfolio performance. The core of the project involves using Facebook's Prophet, a powerful forecasting library, to build a model that predicts future default volumes. The analysis includes visualizing the forecast along with its uncertainty intervals and decomposing the time series into its underlying trend and seasonal components to provide a clear interpretation of the factors driving the predictions.",
      technologies: "Python, Pandas, Prophet, Matplotlib",
      githubLink: "https://github.com/icedarkness/showcase/tree/main/Time-Series%20Forecasting%20of%20Loan%20Volume%20(SARIMA%20vs.%20Prophet)"
    }
  ];

  return (
    <div className="container mt-5">
      <Head>
        <title>Wen Zhang - Data Scientist</title>
        <meta name="description" content="Personal portfolio of Wen Zhang, a Data Scientist specializing in predictive modeling and machine learning." />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="text-center mb-5">
        <h1 className="display-4">Wen Zhang</h1>
        <p className="lead">Data Scientist</p>
        <p>
          <a href="https://www.linkedin.com/in/wenzhangdatascientist/" target="_blank" rel="noopener noreferrer" className="btn btn-outline-primary me-2">LinkedIn</a>
          <a href="http://icedarkness.github.io/" target="_blank" rel="noopener noreferrer" className="btn btn-outline-secondary">GitHub</a>
          <a href="/Wen_Zhang_Principal_Data_Scientist_Resume.pdf" download className="btn btn-outline-success ms-2">Download Resume</a>
        </p>
      </header>

      <main>
        <section id="about" className="mb-5">
          <h2>About Me</h2>
          <p>
            I am a passionate Data Scientist with a deep background in the banking, specialty finance, and fintech sectors. My work focuses on transforming complex data into actionable insights that solve real-world problems. I specialize in developing and deploying robust predictive models, from FCRA-compliant credit risk systems to non-FCRA models for small business lending.
          </p>
          <p>
            I am highly skilled in modernizing legacy systems and automating analytics pipelines to enhance scalability and efficiency. My technical toolkit includes Python, SQL, SAS, and a variety of machine learning frameworks. I hold a Master of Science in Applied Statistics & Operational Research and am a certified Deep Learning specialist. I am driven by the challenge of finding elegant, data-driven solutions that improve decision-making and create a measurable impact.
          </p>
        </section>

        <section id="skills" className="mb-5">
          <h2>Skills</h2>
          <ul>
            <li><strong>Statistical Machine Learning:</strong> GBM (XGBoost), Random Forest, SVM, KNN, AdaBoost, LSTM, Logistic Regression</li>
            <li><strong>Data Analytics:</strong> Data Mining & Validation, Big Data Analytics, ETL Processes, Credit Scoring</li>
            <li><strong>Risk Modeling:</strong> Predictive Risk Modeling, Financial Risk Modeling, Fraud Detection & Prevention</li>
            <li><strong>Optimization Methods:</strong> Stochastic Processes, Linear Programming, Multi-Objective Optimization (NSGA-II)</li>
            <li><strong>Programming & Tools:</strong> Python, SQL, SAS, Angoss</li>
            <li><strong>Certifications:</strong> Deep Learning Specialization</li>
          </ul>
        </section>

        <section id="projects">
          <h2>Projects</h2>
          {projects.map((project, index) => (
            <div className="card mb-3" key={index}>
              <div className="card-body">
                <h5 className="card-title">{project.title}</h5>
                <p className="card-text">{project.description}</p>
                <p className="card-text"><small className="text-muted">{project.technologies}</small></p>
                <a href={project.githubLink} className="btn btn-primary" target="_blank" rel="noopener noreferrer">View on GitHub</a>
              </div>
            </div>
          ))}
        </section>
      </main>

      <footer className="text-center mt-5 py-3">
        <p>&copy; {new Date().getFullYear()} Wen Zhang</p>
      </footer>
    </div>
  );
}