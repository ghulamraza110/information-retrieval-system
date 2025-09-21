"""
Command-line interface for the Information Retrieval System
"""

import os
import sys
from ir_core import InformationRetrievalSystem

class SearchCLI:
    """Command-line interface for IR system"""
    
    def __init__(self):
        self.ir_system = InformationRetrievalSystem()
        self.data_directory = "data"
    
    def load_documents(self):
        """Load documents from the data directory"""
        print("Loading documents...")
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created {self.data_directory} directory. Please add your .txt files there.")
            return False
        
        count = self.ir_system.load_documents_from_directory(self.data_directory)
        
        if count == 0:
            print(f"No documents found in {self.data_directory}/")
            print("Please add some .txt files to the data directory.")
            return False
        
        print(f"Loaded {count} documents successfully!")
        return True
    
    def show_statistics(self):
        """Display system statistics"""
        stats = self.ir_system.get_statistics()
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        print(f"Total Documents: {stats['total_documents']}")
        print(f"Vocabulary Size: {stats['vocabulary_size']}")
        print(f"Index Status: {'Built' if stats['indexed'] else 'Not Built'}")
        print(f"Average Document Length: {stats['average_doc_length']:.1f} words")
        print("="*50)
    
    def search_interactive(self):
        """Interactive search mode"""
        print("\n" + "="*50)
        print("INTERACTIVE SEARCH MODE")
        print("="*50)
        print("Enter your search queries (type 'quit' to exit)")
        print("Commands:")
        print("  - 'stats': Show system statistics")
        print("  - 'list': List all documents")
        print("  - 'view <doc_id>': View document content")
        print("  - 'quit': Exit search mode")
        print("-" * 50)
        
        while True:
            try:
                query = input("\nSearch> ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                elif query.lower() == 'stats':
                    self.show_statistics()
                
                elif query.lower() == 'list':
                    self.list_documents()
                
                elif query.lower().startswith('view '):
                    doc_id = query[5:].strip()
                    self.view_document(doc_id)
                
                elif query:
                    self.perform_search(query)
                
                else:
                    print("Please enter a search query or command.")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def list_documents(self):
        """List all documents in the system"""
        if not self.ir_system.documents:
            print("No documents loaded.")
            return
        
        print("\nAVAILABLE DOCUMENTS:")
        print("-" * 30)
        for doc_id, doc in self.ir_system.documents.items():
            print(f"ID: {doc_id}")
            print(f"Title: {doc.title}")
            print(f"Length: {len(doc.tokens)} words")
            print("-" * 30)
    
    def view_document(self, doc_id: str):
        """View the content of a specific document"""
        content = self.ir_system.get_document_content(doc_id)
        if content:
            doc = self.ir_system.documents[doc_id]
            print(f"\nDOCUMENT: {doc.title}")
            print("=" * 50)
            print(content)
            print("=" * 50)
        else:
            print(f"Document '{doc_id}' not found.")
    
    def perform_search(self, query: str, top_k: int = 10):
        """Perform a search and display results"""
        print(f"\nSearching for: '{query}'")
        print("-" * 50)
        
        results = self.ir_system.search(query, top_k)
        
        if not results:
            print("No matching documents found.")
            return
        
        print(f"Found {len(results)} matching documents:\n")
        
        for i, (doc_id, score, title) in enumerate(results, 1):
            print(f"{i}. {title}")
            print(f"   Document ID: {doc_id}")
            print(f"   Relevance Score: {score:.4f}")
            
            # Show snippet
            content = self.ir_system.get_document_content(doc_id)
            if content:
                snippet = content[:200] + "..." if len(content) > 200 else content
                print(f"   Preview: {snippet}")
            print()
    
    def run(self):
        """Main CLI loop"""
        print("="*60)
        print("INFORMATION RETRIEVAL SYSTEM")
        print("="*60)
        
        # Load documents
        if not self.load_documents():
            return
        
        # Build index
        print("Building search index...")
        self.ir_system.build_index()
        
        # Show statistics
        self.show_statistics()
        
        # Start interactive search
        self.search_interactive()

def main():
    """Entry point for CLI"""
    if len(sys.argv) > 1:
        # Command-line search mode
        query = " ".join(sys.argv[1:])
        cli = SearchCLI()
        
        if cli.load_documents():
            cli.ir_system.build_index()
            cli.perform_search(query)
    else:
        # Interactive mode
        cli = SearchCLI()
        cli.run()

if __name__ == "__main__":
    main()
