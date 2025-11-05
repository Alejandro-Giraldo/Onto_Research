"""
Hetionet Query Agent using LangGraph with Critical Reasoning Loop

This module provides a sophisticated agent for querying the Hetionet biomedical knowledge graph
using LangGraph with a critical reasoning workflow that includes reflection and refinement.
"""

import os
from typing import Dict, List, TypedDict, Annotated
import operator
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


def make_text2cypher_tool(llm: ChatOllama):
    """Create a LangChain tool that converts a question to schema-aware Cypher."""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a text-to-Cypher generator for the Hetionet Neo4j database.
            Follow these rules strictly:
            - Use Hetionet labels when appropriate: Gene, Disease, Compound, Anatomy, Symptom, Pathway, BiologicalProcess, CellularComponent, MolecularFunction, PharmacologicClass.
            - Use relationships where relevant: ASSOCIATES (Gene-Disease), TREATS (Compound-Disease), BINDS (Compound-Gene), PARTICIPATES (Gene-Pathway), CAUSES (Disease-Symptom), EXPRESSES (Gene-Anatomy).
            - Include LIMIT 25 unless the user asks for an aggregate count only.
            - Return only Cypher. No prose.
            """,
        ),
        (
            "system",
            """Schema (if provided):\n{db_schema}""",
        ),
        (
            "human",
            """Question: {question}\n\nOnly output Cypher.""",
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    @tool("text2cypher")
    def text2cypher_tool(question: str, db_schema: str = "") -> str:
        """Convert a natural language biomedical question into a Cypher query for Hetionet."""
        return chain.invoke({"question": question, "db_schema": db_schema})

    return text2cypher_tool


class AgentState(TypedDict):
    """State for the Hetionet query agent"""
    messages: Annotated[List[BaseMessage], operator.add]
    query: str
    cypher_query: str
    graph_response: str
    reflection: str
    final_answer: str
    iteration_count: int
    max_iterations: int


class HetionetAgent:
    """
    A sophisticated agent for querying Hetionet with critical reasoning capabilities.
    
    This agent uses a multi-step reasoning process:
    1. Query Analysis - Understanding the user's question
    2. Cypher Generation - Creating appropriate graph queries
    3. Graph Querying - Executing queries against Hetionet
    4. Response Analysis - Analyzing the results
    5. Critical Reflection - Evaluating and refining the approach
    6. Final Answer - Providing comprehensive response
    """
    
    def __init__(self,
                 graph=None,
                 llm_model: str = "llama3.2",
                 base_url: str = "http://localhost:11434"):
        """
        Initialize the Hetionet agent.
        
        Args:
            graph: Optional Neo4j graph connection (from hetionet_connection.create_hetionet_graph)
            llm_model: LLM model name for Ollama
            base_url: Base URL for Ollama service
        """
        self.llm = ChatOllama(model=llm_model, temperature=0, base_url=base_url)

        # External graph is passed by caller; keep None for offline mode
        self.graph = graph
        
        # Initialize text2cypher tool bound to this LLM
        self.text2cypher_tool = make_text2cypher_tool(self.llm)

        # Initialize the reasoning workflow
        self.workflow = self._create_workflow()

    def render_workflow_graph(self) -> str:
        """Return a Mermaid diagram of the reasoning workflow."""
        mermaid_lines = [
            "graph TD",
            "    A[analyze_query] --> B[generate_cypher]",
            "    B[generate_cypher] --> C[query_graph]",
            "    C[query_graph] --> D[analyze_response]",
            "    D[analyze_response] --> E[critical_reflection]",
            "    E[critical_reflection] -- continue --> B[generate_cypher]",
            "    E[critical_reflection] -- finalize --> F[generate_final_answer]",
            "    F[generate_final_answer] -->|END| END((END))",
        ]
        return "\n".join(mermaid_lines)
    
    def save_workflow_diagram(self, filename: str = "graph.png") -> bool:
        """
        Save the workflow diagram as a PNG file.
        
        Note: This method contains blocking operations and should not be called
        from within an ASGI event loop. Use this method only in standalone scripts
        or when running outside of web servers.
        
        Args:
            filename: Output filename for the PNG diagram
            
        Returns:
            True if successful, False otherwise
        """
        try:
            png_data = self.workflow.get_graph().draw_mermaid_png()
            with open(filename, "wb") as f:
                f.write(png_data)
            return True
        except Exception as e:
            print(f"Warning: Could not generate PNG diagram: {e}")
            return False
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for critical reasoning"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("generate_cypher", self._generate_cypher)
        workflow.add_node("query_graph", self._query_graph)
        workflow.add_node("analyze_response", self._analyze_response)
        workflow.add_node("critical_reflection", self._critical_reflection)
        workflow.add_node("generate_final_answer", self._generate_final_answer)
        
        # Define the flow
        workflow.set_entry_point("analyze_query")
        
        workflow.add_edge("analyze_query", "generate_cypher")
        workflow.add_edge("generate_cypher", "query_graph")
        workflow.add_edge("query_graph", "analyze_response")
        workflow.add_edge("analyze_response", "critical_reflection")
        
        # Conditional edge for reflection loop
        workflow.add_conditional_edges(
            "critical_reflection",
            self._should_continue,
            {
                "continue": "generate_cypher",
                "finalize": "generate_final_answer"
            }
        )
        
        workflow.add_edge("generate_final_answer", END)
        
        app = workflow.compile()
        # Note: PNG generation removed to avoid blocking ASGI event loop
        # Use render_workflow_graph() method for Mermaid diagram if needed
        return app
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user's query to understand intent and requirements"""
        
        query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a biomedical knowledge expert analyzing user queries about Hetionet.
            
            Your task is to:
            1. Understand what the user is asking about
            2. Identify the types of entities they're interested in (genes, diseases, compounds, etc.)
            3. Determine what relationships they want to explore
            4. Assess the complexity of the query
            
            Hetionet contains these main entity types:
            - Gene: Protein-coding human genes
            - Compound: Approved small molecule compounds
            - Disease: Complex diseases
            - Anatomy: Anatomical structures
            - Symptom: Clinical signs and symptoms
            - Side Effect: Adverse drug reactions
            - Biological Process: Molecular activities
            - Cellular Component: Cellular structures
            - Molecular Function: Molecular-level activities
            - Pathway: Molecular pathways
            - Pharmacologic Class: Drug classifications
            
            Provide a clear analysis of the query."""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        analysis_chain = query_analysis_prompt | self.llm | StrOutputParser()
        
        analysis = analysis_chain.invoke({"messages": state["messages"]})
        
        state["messages"].append(AIMessage(content=f"Query Analysis: {analysis}"))
        return state
    
    def _generate_cypher(self, state: AgentState) -> AgentState:
        """Generate schema-aware Cypher for Hetionet from the analyzed question."""

        # Fetch schema text when available to ground the tool
        schema_text = ""
        try:
            if self.graph and getattr(self.graph, "schema", None):
                schema_text = str(self.graph.schema)
        except Exception:
            schema_text = ""

        # Use the LangChain tool to generate Cypher
        cypher_query = self.text2cypher_tool.invoke({
            "question": state.get("query", ""),
            "db_schema": schema_text,
        }).strip()

        state["cypher_query"] = cypher_query
        state["messages"].append(AIMessage(content=f"Generated Cypher Query:\n{cypher_query}"))
        return state
    
    def _query_graph(self, state: AgentState) -> AgentState:
        """Execute the Cypher queries against Hetionet"""
        
        if not self.graph:
            # Offline fallback: return the generated Cypher so user can run elsewhere
            cypher_query = self._extract_cypher_query(state.get("cypher_query", ""))
            state["graph_response"] = f"[offline] Generated Cypher:\n{cypher_query}"
            state["messages"].append(AIMessage(content=state["graph_response"]))
            return state
        
        try:
            cypher_query = self._extract_cypher_query(state["cypher_query"])
            result = self.graph.query(cypher_query)
            
            # Format the result
            if isinstance(result, list) and len(result) > 0:
                formatted_result = self._format_graph_result(result)
            else:
                formatted_result = "No results found for the query."
            
            state["graph_response"] = formatted_result
            state["messages"].append(AIMessage(content=f"Graph Query Results:\n{formatted_result}"))
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            state["graph_response"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    def _analyze_response(self, state: AgentState) -> AgentState:
        """Analyze the graph query results"""
        
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing the results from a Hetionet query.
            
            Your task is to:
            1. Summarize what information was found
            2. Identify key insights and patterns
            3. Assess if the results fully answer the user's question
            4. Note any limitations or gaps in the data
            5. Suggest what additional information might be helpful
            
            Be thorough but concise in your analysis."""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        analysis_chain = analysis_prompt | self.llm | StrOutputParser()
        
        analysis = analysis_chain.invoke({"messages": state["messages"]})
        
        state["messages"].append(AIMessage(content=f"Response Analysis: {analysis}"))
        return state
    
    def _critical_reflection(self, state: AgentState) -> AgentState:
        """Critically reflect on the approach and results"""
        
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical reasoning expert evaluating the query approach and results.
            
            Consider:
            1. Was the query strategy appropriate for the question?
            2. Are the results comprehensive and relevant?
            3. Are there alternative approaches that might yield better results?
            4. Is the current answer sufficient, or should we refine the approach?
            5. What are the limitations of the current results?
            
            Decide whether to:
            - Continue with a refined approach (if results are incomplete)
            - Finalize the answer (if results are satisfactory)
            
            Provide your reasoning and recommendation."""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        reflection_chain = reflection_prompt | self.llm | StrOutputParser()
        
        reflection = reflection_chain.invoke({"messages": state["messages"]})
        
        state["reflection"] = reflection
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        state["messages"].append(AIMessage(content=f"Critical Reflection: {reflection}"))
        
        return state
    
    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue iterating or finalize the answer"""
        
        max_iterations = state.get("max_iterations", 3)
        current_iteration = state.get("iteration_count", 0)
        
        # Check if we've reached max iterations
        if current_iteration >= max_iterations:
            return "finalize"
        
        # Check if the reflection suggests continuing
        reflection = state.get("reflection", "").lower()
        continue_indicators = ["continue", "refine", "try again", "alternative", "incomplete"]
        
        if any(indicator in reflection for indicator in continue_indicators):
            return "continue"
        
        return "finalize"
    
    def _generate_final_answer(self, state: AgentState) -> AgentState:
        """Generate the final comprehensive answer"""
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are providing a final comprehensive answer based on the Hetionet query results and analysis.
            
            Your final answer should:
            1. Directly address the user's original question
            2. Present the key findings clearly and concisely
            3. Include relevant details from the graph data
            4. Acknowledge any limitations or uncertainties
            5. Suggest follow-up questions or related topics if appropriate
            
            Be authoritative but honest about the scope and limitations of the information."""),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        final_chain = final_prompt | self.llm | StrOutputParser()
        
        final_answer = final_chain.invoke({"messages": state["messages"]})
        
        state["final_answer"] = final_answer
        state["messages"].append(AIMessage(content=f"Final Answer: {final_answer}"))
        
        return state
    
    def _extract_cypher_query(self, cypher_text: str) -> str:
        """Extract the actual Cypher query from generated text, stripping code fences."""
        text = cypher_text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = text.strip("`")
            # Remove potential language tag and trailing fences
            text = text.replace("cypher\n", "", 1).replace("\ncypher\n", "\n").strip()
        lines = text.split('\n')
        cypher_lines = []
        in_query = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('MATCH') or line.startswith('CALL') or line.startswith('RETURN'):
                in_query = True
            if in_query and line:
                cypher_lines.append(line)
                if line.endswith(';') or ' LIMIT ' in f" {line} ":
                    break
        
        return '\n'.join(cypher_lines) if cypher_lines else "MATCH (n) RETURN n LIMIT 10"
    
    def _format_graph_result(self, result: List[Dict]) -> str:
        """Format the graph query result for presentation"""
        if not result:
            return "No results found."
        
        formatted = "Query Results:\n"
        for i, record in enumerate(result[:10], 1):  # Limit to first 10 results
            formatted += f"\n{i}. {record}\n"
        
        if len(result) > 10:
            formatted += f"\n... and {len(result) - 10} more results\n"
        
        return formatted
    
    def query(self, question: str, max_iterations: int = 3) -> str:
        """
        Query Hetionet with a question using critical reasoning.
        
        Args:
            question: The question to ask about Hetionet
            max_iterations: Maximum number of reasoning iterations
            
        Returns:
            The final answer to the question
        """
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "query": question,
            "cypher_query": "",
            "graph_response": "",
            "reflection": "",
            "final_answer": "",
            "iteration_count": 0,
            "max_iterations": max_iterations
        }
        
        result = self.workflow.invoke(initial_state)
        return result["final_answer"]


def hetionet_graph(config: RunnableConfig) -> StateGraph:
    """
    Factory function to create a Hetionet agent and return its LangGraph workflow.
    
    This function follows the LangGraph pattern and is required by LangGraph runtime.
    
    Args:
        config: RunnableConfig from LangGraph runtime
        
    Returns:
        Compiled LangGraph workflow
    """
    # Create agent with default settings for LangGraph runtime
    agent = HetionetAgent()
    return agent.workflow


def main():
    """
    Run example queries against Hetionet using the agent.
    """
    # Initialize the agent
    agent = HetionetAgent()
    
    # Example queries
    example_queries = [
        "What genes are associated with osteoarthritis?",
       # "What compounds can treat diabetes?",
       # "What pathways are involved in Alzheimer's disease?",
       # "What are the side effects of aspirin?",
       # "What symptoms are associated with rheumatoid arthritis?"
    ]
    
    print("Hetionet Query Agent - Example Queries")
    print("=" * 50)
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 30)
        
        try:
            answer = agent.query(query)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 50)


# The hetionet_graph function is already defined above and follows LangGraph patterns


if __name__ == "__main__":
    main()
