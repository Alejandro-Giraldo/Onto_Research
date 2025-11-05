"""
Test script for the Hetionet Query Agent

This script provides comprehensive testing and examples for the Hetionet agent,
demonstrating various types of biomedical queries and the critical reasoning workflow.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to import the agent
sys.path.append(str(Path(__file__).parent))

try:
    from .hetionet_agent import HetionetAgent
    from .hetionet_connection import create_hetionet_graph
except ImportError:
    from hetionet_agent import HetionetAgent
    from hetionet_connection import create_hetionet_graph


class HetionetAgentTester:
    """Test suite for the Hetionet Agent"""
    
    def __init__(self, connect: bool = False):
        graph = create_hetionet_graph() if connect else None
        self.agent = HetionetAgent(graph=graph)
        self.test_results = []
    
    def run_basic_tests(self):
        """Run basic functionality tests"""
        print("Running Basic Tests")
        print("=" * 50)
        
        basic_queries = [
            "What is Hetionet?",
            "How many genes are in Hetionet?",
            "What types of entities are in Hetionet?"
        ]
        
        for query in basic_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=2)
                print(f"Result: {result}")
                self.test_results.append(("basic", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("basic", query, "error", str(e)))
    
    def run_gene_disease_tests(self):
        """Test gene-disease association queries"""
        print("\n\nRunning Gene-Disease Association Tests")
        print("=" * 50)
        
        gene_disease_queries = [
            "What genes are associated with osteoarthritis?",
            "What diseases are associated with the BRCA1 gene?",
            "What genes are involved in diabetes?",
            "What is the relationship between APOE and Alzheimer's disease?"
        ]
        
        for query in gene_disease_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=3)
                print(f"Result: {result}")
                self.test_results.append(("gene_disease", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("gene_disease", query, "error", str(e)))
    
    def run_drug_tests(self):
        """Test drug and compound queries"""
        print("\n\nRunning Drug and Compound Tests")
        print("=" * 50)
        
        drug_queries = [
            "What compounds can treat diabetes?",
            "What are the side effects of aspirin?",
            "What drugs are used for hypertension?",
            "What is the mechanism of action of metformin?"
        ]
        
        for query in drug_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=3)
                print(f"Result: {result}")
                self.test_results.append(("drug", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("drug", query, "error", str(e)))
    
    def run_pathway_tests(self):
        """Test pathway and biological process queries"""
        print("\n\nRunning Pathway and Biological Process Tests")
        print("=" * 50)
        
        pathway_queries = [
            "What pathways are involved in cancer?",
            "What biological processes are associated with inflammation?",
            "What genes participate in the insulin signaling pathway?",
            "What pathways are related to cell death?"
        ]
        
        for query in pathway_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=3)
                print(f"Result: {result}")
                self.test_results.append(("pathway", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("pathway", query, "error", str(e)))
    
    def run_complex_queries(self):
        """Test complex multi-hop queries"""
        print("\n\nRunning Complex Multi-hop Queries")
        print("=" * 50)
        
        complex_queries = [
            "What drugs can treat diseases that are associated with genes in the insulin pathway?",
            "What symptoms are associated with diseases that can be treated by aspirin?",
            "What anatomical structures are involved in diseases associated with the APOE gene?",
            "What are the side effects of drugs that treat diseases related to inflammation pathways?"
        ]
        
        for query in complex_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=4)
                print(f"Result: {result}")
                self.test_results.append(("complex", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("complex", query, "error", str(e)))
    
    def run_critical_reasoning_tests(self):
        """Test the critical reasoning capabilities"""
        print("\n\nRunning Critical Reasoning Tests")
        print("=" * 50)
        
        reasoning_queries = [
            "Why might some genes be more associated with certain diseases than others?",
            "What are the limitations of current drug-disease associations in Hetionet?",
            "How reliable are the gene-disease associations in Hetionet?",
            "What factors might influence the strength of drug-target interactions?"
        ]
        
        for query in reasoning_queries:
            print(f"\nQuery: {query}")
            print("-" * 30)
            try:
                result = self.agent.query(query, max_iterations=3)
                print(f"Result: {result}")
                self.test_results.append(("reasoning", query, "success", result))
            except Exception as e:
                print(f"Error: {e}")
                self.test_results.append(("reasoning", query, "error", str(e)))
    
    def generate_test_report(self):
        """Generate a comprehensive test report"""
        print("\n\nTest Report")
        print("=" * 50)
        
        # Count results by category
        categories = {}
        for category, query, status, result in self.test_results:
            if category not in categories:
                categories[category] = {"success": 0, "error": 0, "total": 0}
            categories[category][status] += 1
            categories[category]["total"] += 1
        
        # Print summary
        for category, stats in categories.items():
            success_rate = (stats["success"] / stats["total"]) * 100
            print(f"\n{category.upper()}:")
            print(f"  Total tests: {stats['total']}")
            print(f"  Successful: {stats['success']}")
            print(f"  Errors: {stats['error']}")
            print(f"  Success rate: {success_rate:.1f}%")
        
        # Overall summary
        total_tests = len(self.test_results)
        total_success = sum(1 for _, _, status, _ in self.test_results if status == "success")
        overall_success_rate = (total_success / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\nOVERALL SUMMARY:")
        print(f"  Total tests: {total_tests}")
        print(f"  Successful: {total_success}")
        print(f"  Overall success rate: {overall_success_rate:.1f}%")
    
    def run_all_tests(self):
        """Run all test categories"""
        print("Hetionet Agent Comprehensive Test Suite")
        print("=" * 60)
        
        self.run_basic_tests()
        self.run_gene_disease_tests()
        self.run_drug_tests()
        self.run_pathway_tests()
        self.run_complex_queries()
        self.run_critical_reasoning_tests()
        
        self.generate_test_report()


def interactive_demo(connect: bool = False, print_graph: bool = False):
    """Interactive demo for testing specific queries"""
    print("Hetionet Agent Interactive Demo")
    print("=" * 40)
    print("Enter your questions about biomedical knowledge.")
    print("Type 'quit' to exit, 'help' for example queries.")
    print()
    
    graph = create_hetionet_graph() if connect else None
    agent = HetionetAgent(graph=graph)
    if print_graph:
        print("\nLangGraph workflow (Mermaid):\n")
        print(agent.render_workflow_graph())
    
    example_queries = [
        "What genes are associated with osteoarthritis?",
        "What compounds can treat diabetes?",
        "What pathways are involved in Alzheimer's disease?",
        "What are the side effects of aspirin?",
        "What symptoms are associated with rheumatoid arthritis?",
        "What drugs can treat diseases related to inflammation?",
        "What anatomical structures are involved in cardiovascular disease?",
        "What biological processes are associated with cancer?",
        "What genes participate in the insulin signaling pathway?",
        "What are the limitations of current drug-disease associations?"
    ]
    
    while True:
        try:
            query = input("\nEnter your question: ").strip()
            
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            elif query.lower() == 'help':
                print("\nExample queries:")
                for i, example in enumerate(example_queries, 1):
                    print(f"{i}. {example}")
                continue
            elif not query:
                continue
            
            print(f"\nProcessing: {query}")
            print("-" * 40)
            
            result = agent.query(query, max_iterations=3)
            print(f"\nAnswer: {result}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function to run tests or interactive demo"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Hetionet Agent")
    parser.add_argument("--mode", choices=["test", "interactive"], default="test",
                       help="Run mode: 'test' for automated tests, 'interactive' for demo")
    parser.add_argument("--category", choices=["basic", "gene_disease", "drug", "pathway", "complex", "reasoning"],
                       help="Run specific test category")
    parser.add_argument("--connect", action="store_true", help="Connect to Hetionet")
    parser.add_argument("--print-graph", action="store_true", help="Print LangGraph workflow Mermaid diagram")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        interactive_demo(connect=args.connect, print_graph=args.print_graph)
    else:
        tester = HetionetAgentTester(connect=args.connect)
        
        if args.category:
            # Run specific category
            method_name = f"run_{args.category}_tests"
            if hasattr(tester, method_name):
                getattr(tester, method_name)()
                tester.generate_test_report()
            else:
                print(f"Unknown category: {args.category}")
        else:
            # Run all tests
            tester.run_all_tests()


if __name__ == "__main__":
    main()
