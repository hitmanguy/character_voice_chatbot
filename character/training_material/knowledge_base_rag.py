"""
🔍 RAG KNOWLEDGE BASE FOR IRON MAN CHATBOT
===========================================
Retrieval-Augmented Generation system using ChromaDB for factual accuracy.
Stores MCU canon facts, timeline, tech specs, relationships for retrieval during inference.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️  ChromaDB not available. Install: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  SentenceTransformers not available. Install: pip install sentence-transformers")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IronManKnowledgeBase:
    """
    RAG system for Iron Man chatbot with factual knowledge retrieval.
    Uses vector similarity search to inject relevant context into prompts.
    """
    
    def __init__(self, 
                 config_path: str = "persona_config.json",
                 db_path: str = "./ironman_chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize knowledge base with vector store."""
        self.config_path = config_path
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available. Install: pip install chromadb")
            self.db = None
            self.collection = None
            return
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name="ironman_knowledge")
            logger.info(f"✅ Loaded existing knowledge base with {self.collection.count()} documents")
        except:
            self.collection = self.client.create_collection(
                name="ironman_knowledge",
                metadata={"description": "Iron Man factual knowledge base"}
            )
            logger.info("✅ Created new knowledge base collection")
        
        # Load persona config
        self.config = self._load_config(config_path)
    
    def _load_config(self, path: str) -> Dict:
        """Load persona configuration."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _generate_doc_id(self, text: str) -> str:
        """Generate unique ID for document."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def populate_from_config(self):
        """Extract and index all factual knowledge from persona config."""
        logger.info("📚 Populating knowledge base from persona config...")
        
        if not self.config:
            logger.error("No config loaded!")
            return
        
        factual_base = self.config.get("factual_knowledge_base", {})
        documents = []
        metadatas = []
        ids = []
        
        # 1. Personal history facts
        personal = factual_base.get("personal_history", {})
        if personal:
            # Birth and education
            doc = f"Tony Stark was born in {personal.get('birth_year', '1970')}. His parents were {personal.get('parents', 'Howard and Maria Stark')}. He graduated from MIT at age 17, showcasing his genius-level intellect early."
            documents.append(doc)
            metadatas.append({"category": "personal_history", "subcategory": "early_life"})
            ids.append(self._generate_doc_id(doc))
            
            # Company
            doc = f"Tony Stark inherited and later reformed Stark Industries, a major technology and defense company."
            documents.append(doc)
            metadatas.append({"category": "personal_history", "subcategory": "company"})
            ids.append(self._generate_doc_id(doc))
            
            # Major events
            for event in personal.get("major_events", []):
                documents.append(event)
                metadatas.append({"category": "personal_history", "subcategory": "major_events"})
                ids.append(self._generate_doc_id(event))
        
        # 2. Suit progression
        for suit in factual_base.get("suit_progression", []):
            documents.append(suit)
            metadatas.append({"category": "technology", "subcategory": "iron_man_suits"})
            ids.append(self._generate_doc_id(suit))
        
        # 3. Key relationships
        relationships = factual_base.get("key_relationships", {})
        for person, description in relationships.items():
            doc = f"{person}: {description}"
            documents.append(doc)
            metadatas.append({"category": "relationships", "subcategory": person.lower().replace(' ', '_')})
            ids.append(self._generate_doc_id(doc))
        
        # 4. Tech innovations
        for tech in factual_base.get("tech_innovations", []):
            documents.append(tech)
            metadatas.append({"category": "technology", "subcategory": "innovations"})
            ids.append(self._generate_doc_id(tech))
        
        # 5. Core personality traits (for consistency)
        core = self.config.get("core_personality", {})
        for trait in core.get("primary_traits", []):
            documents.append(f"Tony Stark personality trait: {trait}")
            metadatas.append({"category": "personality", "subcategory": "core_traits"})
            ids.append(self._generate_doc_id(trait))
        
        # 6. Speech patterns and catchphrases
        speech = core.get("speech_patterns", {})
        for phrase in speech.get("catchphrases", []):
            doc = f"Tony Stark catchphrase: {phrase}"
            documents.append(doc)
            metadatas.append({"category": "personality", "subcategory": "catchphrases"})
            ids.append(self._generate_doc_id(doc))
        
        # Add to collection
        if documents:
            # Remove duplicates
            unique_docs = []
            unique_ids = []
            unique_meta = []
            seen = set()
            
            for doc, doc_id, meta in zip(documents, ids, metadatas):
                if doc_id not in seen:
                    unique_docs.append(doc)
                    unique_ids.append(doc_id)
                    unique_meta.append(meta)
                    seen.add(doc_id)
            
            self.collection.add(
                documents=unique_docs,
                metadatas=unique_meta,
                ids=unique_ids
            )
            logger.info(f"✅ Indexed {len(unique_docs)} knowledge documents")
        
        return len(documents)
    
    def add_custom_facts(self, facts: List[Dict[str, str]]):
        """
        Add custom factual knowledge.
        
        Args:
            facts: List of dicts with 'text', 'category', 'subcategory'
        """
        logger.info(f"📝 Adding {len(facts)} custom facts...")
        
        documents = []
        metadatas = []
        ids = []
        
        for fact in facts:
            text = fact.get("text", "")
            if text:
                documents.append(text)
                metadatas.append({
                    "category": fact.get("category", "custom"),
                    "subcategory": fact.get("subcategory", "general")
                })
                ids.append(self._generate_doc_id(text))
        
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"✅ Added {len(documents)} custom facts")
    
    def retrieve(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve most relevant knowledge for a query.
        
        Args:
            query: User's question or input
            top_k: Number of documents to retrieve
            category_filter: Optional category filter
            
        Returns:
            List of relevant documents with metadata
        """
        if not self.collection:
            logger.warning("No collection available")
            return []
        
        try:
            # Build filter if needed
            where_filter = None
            if category_filter:
                where_filter = {"category": category_filter}
            
            # Query
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            # Format results
            retrieved = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    retrieved.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results.get('distances') else None
                    })
            
            return retrieved
        
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def build_context_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Build enriched prompt with retrieved context.
        
        Args:
            query: User's input
            top_k: Number of context documents
            
        Returns:
            Context string to prepend to prompt
        """
        retrieved = self.retrieve(query, top_k=top_k)
        
        if not retrieved:
            return ""
        
        context_parts = ["**RELEVANT KNOWLEDGE:**"]
        for i, doc in enumerate(retrieved, 1):
            context_parts.append(f"{i}. {doc['text']}")
        
        context_parts.append("\nUse this information if relevant to answer accurately.\n")
        return "\n".join(context_parts)
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        if not self.collection:
            return {"error": "No collection"}
        
        count = self.collection.count()
        
        # Get category breakdown
        all_docs = self.collection.get()
        categories = {}
        if all_docs and all_docs['metadatas']:
            for meta in all_docs['metadatas']:
                cat = meta.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "total_documents": count,
            "categories": categories,
            "db_path": self.db_path
        }
    
    def search_by_category(self, category: str, limit: int = 10) -> List[str]:
        """Get all documents in a category."""
        if not self.collection:
            return []
        
        try:
            results = self.collection.get(
                where={"category": category},
                limit=limit
            )
            return results['documents'] if results else []
        except Exception as e:
            logger.error(f"Category search error: {e}")
            return []
    
    def export_knowledge_base(self, output_path: str = "knowledge_base_export.json"):
        """Export entire knowledge base to JSON."""
        logger.info(f"💾 Exporting knowledge base to {output_path}...")
        
        if not self.collection:
            logger.error("No collection to export")
            return
        
        all_docs = self.collection.get()
        
        export_data = {
            "total_documents": len(all_docs['documents']) if all_docs else 0,
            "knowledge": []
        }
        
        if all_docs and all_docs['documents']:
            for i, doc in enumerate(all_docs['documents']):
                export_data["knowledge"].append({
                    "id": all_docs['ids'][i] if all_docs['ids'] else None,
                    "text": doc,
                    "metadata": all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Exported {export_data['total_documents']} documents")
    
    def clear_all(self):
        """Clear entire knowledge base (use with caution)."""
        logger.warning("⚠️  Clearing entire knowledge base...")
        if self.collection:
            self.client.delete_collection("ironman_knowledge")
            self.collection = self.client.create_collection(
                name="ironman_knowledge",
                metadata={"description": "Iron Man factual knowledge base"}
            )
            logger.info("✅ Knowledge base cleared")


def main():
    """Main execution for setup and testing."""
    print("=" * 60)
    print("🔍 IRON MAN KNOWLEDGE BASE SETUP")
    print("=" * 60)
    print()
    
    if not CHROMADB_AVAILABLE:
        print("❌ ChromaDB not installed!")
        print("   Install: pip install chromadb")
        return
    
    # Initialize
    kb = IronManKnowledgeBase()
    
    # Populate from config
    print("📚 Populating knowledge base from persona config...")
    kb.populate_from_config()
    
    # Add some additional custom facts
    print("📝 Adding custom facts...")
    custom_facts = [
        {
            "text": "The Mark 50 suit (Infinity War) uses bleeding-edge nanotech stored in the arc reactor housing unit on Tony's chest.",
            "category": "technology",
            "subcategory": "nanotech"
        },
        {
            "text": "Tony Stark created the B.A.R.F. (Binarily Augmented Retro-Framing) technology for therapeutic memory reconstruction.",
            "category": "technology",
            "subcategory": "medical_tech"
        },
        {
            "text": "The Iron Spider suit given to Peter Parker includes instant-kill mode, reconnaissance drone, and web wing gliders.",
            "category": "technology",
            "subcategory": "spider_man_tech"
        },
        {
            "text": "Tony's workshop in his Malibu mansion was destroyed by the Mandarin's attack in 2013.",
            "category": "personal_history",
            "subcategory": "locations"
        },
        {
            "text": "Morgan Stark is Tony and Pepper's daughter, born after the Snap.",
            "category": "relationships",
            "subcategory": "family"
        }
    ]
    kb.add_custom_facts(custom_facts)
    
    # Show stats
    print("\n📊 Knowledge Base Statistics:")
    stats = kb.get_stats()
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Categories: {stats['categories']}")
    
    # Test retrieval
    print("\n🧪 Testing retrieval...")
    test_queries = [
        "Tell me about the arc reactor",
        "What suits did Tony use in major battles?",
        "Who is Pepper Potts?",
        "When was Tony born?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = kb.retrieve(query, top_k=2)
        for i, result in enumerate(results, 1):
            print(f"      {i}. {result['text'][:100]}...")
    
    # Export
    print("\n💾 Exporting knowledge base...")
    kb.export_knowledge_base("ironman_knowledge_export.json")
    
    print("\n✅ Knowledge base setup complete!")
    print(f"📁 Database stored at: {kb.db_path}")
    print("\n📝 Next steps:")
    print("   1. Review exported knowledge: ironman_knowledge_export.json")
    print("   2. Add more custom facts if needed")
    print("   3. Use in ironman_pro.py for RAG-enhanced responses")
    print()


if __name__ == "__main__":
    main()
