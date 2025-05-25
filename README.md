# GraphRAG_ChatBot

This is my final submission for the **Technical Test: AI-Based Chatbot Development**.

---

## Running the Chatbot Locally

To run the chatbot locally:

### 1. Clone the Repository

```bash
gh repo clone Davdioc/Nestle_ChatBot
```

### 2. Add Environment Variables

Create a `.env` file in **both the root** and the **backend folder**. Here’s a sample template:

```env
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_EMBEDDINGS_API=
AZURE_OPENAI_EMBEDDINGS_ENDPOINT=
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-large
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 3. Run the App

```bash
docker compose up --build
```

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- Neo4j: `http://localhost:7474`

---

## Notes

- The local version uses a local Neo4j database.
- The deployed version uses **Neo4j Aura**, and its backend has been optimized using asynchronous programming and multi-threading for better performance.
- Both versions share the same underlying logic and implementation.

---

## Technologies Used

- **Azure OpenAI (GPT-4o)** – Transforms processed documents into graph documents.
- **Azure OpenAI Embeddings (text-embedding-3-large)** – Converts the graph database into a Neo4j vector store.
- **ScraperAPI** – Bypasses `madewithnestle.ca`'s anti-bot protections for web crawling.
- **Neo4j Aura** – Hosts the graph and vector database in production.
- **FastAPI** – Powers the backend server.
- **React + Vite** – Builds the frontend interface.
- **Docker + Docker Compose** – Containerizes the app for local deployment.
- **Azure Container Registry** – Stores the containerized backend image.
- **Azure Web App** – Hosts the backend service.
- **Azure Static Web Apps** – Hosts the frontend.
- **GitHub Actions** – Automates frontend CI/CD deployments.

---

## Known Limitations

1. **No dynamic scraping:** Due to resource constraints, the system doesn’t scrape `madewithnestle.ca` in real time.
2. **Partial knowledge base:** The site contains ~1820 pages (from `sitemap.xml`), but only 82 key pages were processed due to limited resources.
3. **Frontend responsiveness:** The layout may not respond well on screen sizes narrower than **1041px**.

---

## Final Note

Thank you for the opportunity to interview with such a prestigious company. This project was both a challenge and a pleasure to build.

— *David Oche*
