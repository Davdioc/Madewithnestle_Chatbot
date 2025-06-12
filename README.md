# GraphRAG_ChatBot

[Jump to Second Iteration Improvements](#second-iteration-improvements)
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
- The deployed version uses **Neo4j Aura**, and its backend has been optimized using asynchronous programming and concurrent request handling for better performance.
- Both versions share the same underlying logic and implementation.

---

## Technologies Used

- **Azure OpenAI (GPT-4o)** – Transforms processed documents into graph documents.
- **Azure OpenAI Embeddings (text-embedding-3-large)** – Converts the graph database into a Neo4j vector store.
- **ScraperAPI** – Bypasses `madewithnestle.ca`'s anti-bot protections for web crawling and scraping.
- **Neo4j Aura** – Hosts the graph and vector database in production.
- **FastAPI** – Powers the backend server.
- **React + Vite** – Builds the frontend interface.
- **Docker + Docker Compose** – Containerizes the app for local deployment.
- **Azure Container Registry** – Stores the containerized backend image.
- **Azure Web App** – Hosts the backend service.
- **Azure Static Web Apps** – Hosts the frontend.
- **GitHub Actions** – Automates frontend CI/CD deployments.

---

## Known Limitations and Additional Features

1. **No dynamic scraping:** Due to resource constraints, the system doesn’t scrape `madewithnestle.ca` in real time.
2. **Partial knowledge base:** The site contains ~1820 pages (from `/sitemap.xml`), only pages from `/sitmap` (82) were processed due to limited resources.
3. **Frontend responsiveness:** The layout may not respond well on screen sizes narrower than **1041px**.

- **Additional features can be found in the `/documentation` directory**
---

## Second Iteration Improvements

As part of a second iteration to address structured query limitations and improve system resilience, the following enhancements were implemented:

### Structured Query Support (Mocked)

- Embedded metadata and a simplified index were used to allow the chatbot to respond to questions like:
  - “How many Nestlé products are listed on the site?”
  - “How many products are under the coffee category?”
- These queries are routed internally to a logic layer that interprets and retrieves structured counts from indexed or embedded metadata.

### Store Locator + Amazon Integration

- If the user asks where to buy a product:
  - The chatbot uses `navigator.geolocation` to fetch nearby stores using the Google Maps API.
  - Each result includes name, address, distance, open/closed status, and a map link.
  - A fallback Amazon link for the product is also provided.

### UI/UX Enhancements

- Typing animation and live scroll-to-latest message behavior.
- Mobile-optimized chat layout with aesthetic adjustments like:
  - Bubble shadows
  - Rounded corners
  - Fade-in transitions
- Brand-matching color scheme, clean message alignment, and multilingual preview support.
- "Suggested Questions" can now be toggled via a persistent FAQ button and fade out smoothly after use.
---

- **Additional features can be found in the [`/documentation`](https://github.com/Davdioc/Madewithnestle_Chatbot/tree/main/documentation2) directory**

