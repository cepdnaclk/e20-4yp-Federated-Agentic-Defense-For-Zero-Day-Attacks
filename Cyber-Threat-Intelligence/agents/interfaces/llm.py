"""
LLM Implementations for Threat Analysis.

This module provides concrete LLM implementations for generating
human-readable threat analysis and reasoning summaries.
"""

import logging
from typing import List, Dict, Any, Optional

from agents.interfaces.base import LLMInterface, LLMResponse

logger = logging.getLogger(__name__)


class OllamaLLM(LLMInterface):
    """
    Ollama LLM implementation for local model inference.
    
    Supports running local models like Llama 3, Mistral, etc.
    through the Ollama API.
    
    Features:
        - Local inference (no API costs)
        - Multiple model support
        - RAG-optimized prompting
        - Streaming support
    
    Example:
        >>> llm = OllamaLLM(model="llama3")
        >>> response = llm.generate("Explain this network anomaly")
        >>> print(response.content)
    """
    
    # Threat analysis system prompt
    DEFAULT_SYSTEM_PROMPT = """You are a cybersecurity expert specializing in network intrusion detection and threat analysis. Your role is to analyze network traffic anomalies and provide clear, actionable insights.

When analyzing threats:
1. Identify the likely attack type based on the data provided
2. Explain the potential impact and severity
3. Suggest immediate mitigation steps
4. Reference similar historical attacks when available

Be concise but thorough. Use technical terminology appropriately."""
    
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        """
        Initializes the Ollama LLM client.
        
        Args:
            model: Ollama model name (e.g., 'llama3', 'mistral', 'codellama').
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
        """
        self._model = model
        self._base_url = base_url
        self._timeout = timeout
        
        logger.info("OllamaLLM initialized: model=%s, url=%s", model, base_url)
    
    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates text using Ollama.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions (uses default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional Ollama parameters.
        
        Returns:
            LLMResponse with generated content.
        """
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError(
                "langchain-community required. Install with: "
                "pip install langchain-community"
            )
        
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
        # Create Ollama instance
        llm = Ollama(
            model=self._model,
            base_url=self._base_url,
            temperature=temperature,
            num_predict=max_tokens,
            timeout=self._timeout,
        )
        
        # Construct full prompt with system message
        full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Generate response
        try:
            response = llm.invoke(full_prompt)
            
            return LLMResponse(
                content=response.strip(),
                model=self._model,
                tokens_used=None,  # Ollama doesn't return token counts easily
                metadata={"temperature": temperature, "max_tokens": max_tokens},
            )
        except Exception as e:
            logger.error("Ollama generation failed: %s", str(e))
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User query about the threat.
            context: List of retrieved context documents.
            system_prompt: Optional custom system prompt.
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse with analysis.
        """
        # Build context string
        context_str = "\n\n---\n\n".join([
            f"[Reference {i+1}]: {ctx}" for i, ctx in enumerate(context)
        ])
        
        # Build RAG prompt
        rag_prompt = f"""Analyze the following network threat using the provided reference information.

## Retrieved Context (Similar Historical Attacks):
{context_str}

## Current Threat Query:
{query}

## Analysis Required:
Based on the context above and the current threat data, provide:
1. Threat Classification: What type of attack is this most likely?
2. Similarity to Known Attacks: How does this compare to the historical attacks?
3. Severity Assessment: High/Medium/Low and why
4. Recommended Actions: What should the security team do?
5. Key Indicators: What specific indicators support your analysis?

Provide a clear, structured analysis:"""
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class MockLLM(LLMInterface):
    """
    Mock LLM for testing without requiring actual LLM infrastructure.
    
    Returns template responses useful for development and testing.
    """
    
    def __init__(self, model_name: str = "mock-llm"):
        """Initializes the mock LLM."""
        self._model_name = model_name
        logger.info("MockLLM initialized (for testing)")
    
    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model_name
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """Returns a template response."""
        template = f"""## Threat Analysis (Mock Response)

Based on the provided information, this appears to be a potential security threat.

**Analysis:**
- The network traffic patterns indicate anomalous behavior
- Further investigation is recommended
- Standard incident response procedures should be followed

**Recommendations:**
1. Isolate affected systems if necessary
2. Collect additional logs for forensic analysis
3. Update detection signatures based on observed patterns

*Note: This is a mock response for testing purposes.*
"""
        return LLMResponse(
            content=template,
            model=self._model_name,
            tokens_used=len(template.split()),
            metadata={"mock": True},
        )
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Returns a template RAG response."""
        context_summary = f"Analyzed {len(context)} similar historical attacks."
        
        template = f"""## Threat Analysis with Context (Mock Response)

{context_summary}

**Query:** {query[:200]}...

**Similar Attack Patterns Found:**
The retrieved context shows patterns consistent with known attack categories.

**Classification:** Potential threat requiring further analysis

**Severity:** Medium (based on pattern similarity)

**Recommended Actions:**
1. Review the specific indicators mentioned
2. Cross-reference with current threat intelligence
3. Implement temporary mitigation if risk is elevated

*Note: This is a mock response for testing purposes.*
"""
        return LLMResponse(
            content=template,
            model=self._model_name,
            tokens_used=len(template.split()),
            metadata={"mock": True, "context_count": len(context)},
        )


class OpenAILLM(LLMInterface):
    """
    OpenAI LLM implementation for GPT-3.5/GPT-4 inference.
    
    Uses the OpenAI API for high-quality threat analysis.
    
    Features:
        - GPT-3.5-turbo and GPT-4 support
        - Structured chat completions
        - Token usage tracking
        - RAG-optimized prompting
    
    Example:
        >>> llm = OpenAILLM(api_key="sk-...", model="gpt-4")
        >>> response = llm.generate("Analyze this network anomaly")
        >>> print(response.content)
    
    Environment Variable:
        Set OPENAI_API_KEY to avoid passing api_key directly.
    """
    
    # Threat analysis system prompt (same as Ollama for consistency)
    DEFAULT_SYSTEM_PROMPT = """You are a cybersecurity expert specializing in network intrusion detection and threat analysis. Your role is to analyze network traffic anomalies and provide clear, actionable insights.

When analyzing threats:
1. Identify the likely attack type based on the data provided
2. Explain the potential impact and severity
3. Suggest immediate mitigation steps
4. Reference similar historical attacks when available

Be concise but thorough. Use technical terminology appropriately."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Initializes the OpenAI LLM client.
        
        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: Model name ('gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', etc.).
            organization: Optional OpenAI organization ID.
            base_url: Optional custom API base URL (for Azure OpenAI, etc.).
            timeout: Request timeout in seconds.
        """
        import os
        
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Pass api_key or set OPENAI_API_KEY env var."
            )
        
        self._model = model
        self._organization = organization
        self._base_url = base_url
        self._timeout = timeout
        self._client = None  # Lazy initialization
        
        logger.info("OpenAILLM initialized: model=%s", model)
    
    def _get_client(self):
        """Lazily initializes the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
            
            self._client = OpenAI(
                api_key=self._api_key,
                organization=self._organization,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._client
    
    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates text using OpenAI Chat Completions API.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions (uses default if None).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional OpenAI parameters (top_p, presence_penalty, etc.).
        
        Returns:
            LLMResponse with generated content.
        """
        client = self._get_client()
        
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
        # Build messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            # Extract response details
            choice = response.choices[0]
            content = choice.message.content.strip()
            
            # Calculate token usage
            tokens_used = None
            if response.usage:
                tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                model=self._model,
                tokens_used=tokens_used,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                },
            )
        except Exception as e:
            logger.error("OpenAI generation failed: %s", str(e))
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User query about the threat.
            context: List of retrieved context documents.
            system_prompt: Optional custom system prompt.
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse with analysis.
        """
        # Build context string
        context_str = "\n\n---\n\n".join([
            f"[Reference {i+1}]: {ctx}" for i, ctx in enumerate(context)
        ])
        
        # Build RAG prompt
        rag_prompt = f"""Analyze the following network threat using the provided reference information.

## Retrieved Context (Similar Historical Attacks):
{context_str}

## Current Threat Query:
{query}

## Analysis Required:
Based on the context above and the current threat data, provide:
1. Threat Classification: What type of attack is this most likely?
2. Similarity to Known Attacks: How does this compare to the historical attacks?
3. Severity Assessment: High/Medium/Low and why
4. Recommended Actions: What should the security team do?
5. Key Indicators: What specific indicators support your analysis?

Provide a clear, structured analysis:"""
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )


class GroqLLM(LLMInterface):
    """
    Groq LLM implementation for fast, FREE inference.
    
    Groq provides free API access to models like Llama 3, Mixtral, etc.
    with extremely fast inference speeds.
    
    Features:
        - FREE API tier (generous rate limits)
        - Ultra-fast inference (LPU technology)
        - Multiple model support (Llama 3, Mixtral, Gemma)
        - OpenAI-compatible API
    
    Example:
        >>> llm = GroqLLM(api_key="gsk_...", model="llama-3.3-70b-versatile")
        >>> response = llm.generate("Analyze this network anomaly")
        >>> print(response.content)
    
    Environment Variable:
        Set GROQ_API_KEY to avoid passing api_key directly.
    
    Get free API key at: https://console.groq.com/keys
    """
    
    # Threat analysis system prompt (same as others for consistency)
    DEFAULT_SYSTEM_PROMPT = """You are a cybersecurity expert specializing in network intrusion detection and threat analysis. Your role is to analyze network traffic anomalies and provide clear, actionable insights.

When analyzing threats:
1. Identify the likely attack type based on the data provided
2. Explain the potential impact and severity
3. Suggest immediate mitigation steps
4. Reference similar historical attacks when available

Be concise but thorough. Use technical terminology appropriately."""
    
    # Available Groq models (as of 2024)
    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",      # Best quality, free
        "llama-3.1-8b-instant",          # Fast, free
        "llama3-70b-8192",               # Legacy Llama 3
        "llama3-8b-8192",                # Legacy Llama 3 small
        "mixtral-8x7b-32768",            # Mixtral, 32k context
        "gemma2-9b-it",                  # Google Gemma 2
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: int = 60,
    ):
        """
        Initializes the Groq LLM client.
        
        Args:
            api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
            model: Model name (default: llama-3.3-70b-versatile).
            timeout: Request timeout in seconds.
        """
        import os
        
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Groq API key required. Pass api_key or set GROQ_API_KEY env var. "
                "Get free key at: https://console.groq.com/keys"
            )
        
        self._model = model
        self._timeout = timeout
        self._client = None  # Lazy initialization
        
        logger.info("GroqLLM initialized: model=%s (FREE tier)", model)
    
    def _get_client(self):
        """Lazily initializes the Groq client."""
        if self._client is None:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "groq package required. Install with: pip install groq"
                )
            
            self._client = Groq(
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self._client
    
    @property
    def model_name(self) -> str:
        """Returns the model name."""
        return self._model
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates text using Groq API.
        
        Args:
            prompt: User prompt.
            system_prompt: System instructions (uses default if None).
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional Groq parameters.
        
        Returns:
            LLMResponse with generated content.
        """
        client = self._get_client()
        
        if system_prompt is None:
            system_prompt = self.DEFAULT_SYSTEM_PROMPT
        
        # Build messages for chat completion
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            
            # Extract response details
            choice = response.choices[0]
            content = choice.message.content.strip()
            
            # Calculate token usage
            tokens_used = None
            if response.usage:
                tokens_used = response.usage.total_tokens
            
            return LLMResponse(
                content=content,
                model=self._model,
                tokens_used=tokens_used,
                metadata={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "finish_reason": choice.finish_reason,
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                    "provider": "groq",
                },
            )
        except Exception as e:
            logger.error("Groq generation failed: %s", str(e))
            raise
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generates response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User query about the threat.
            context: List of retrieved context documents.
            system_prompt: Optional custom system prompt.
            **kwargs: Additional generation parameters.
        
        Returns:
            LLMResponse with analysis.
        """
        # Build context string
        context_str = "\n\n---\n\n".join([
            f"[Reference {i+1}]: {ctx}" for i, ctx in enumerate(context)
        ])
        
        # Build RAG prompt
        rag_prompt = f"""Analyze the following network threat using the provided reference information.

## Retrieved Context (Similar Historical Attacks):
{context_str}

## Current Threat Query:
{query}

## Analysis Required:
Based on the context above and the current threat data, provide:
1. Threat Classification: What type of attack is this most likely?
2. Similarity to Known Attacks: How does this compare to the historical attacks?
3. Severity Assessment: High/Medium/Low and why
4. Recommended Actions: What should the security team do?
5. Key Indicators: What specific indicators support your analysis?

Provide a clear, structured analysis:"""
        
        return self.generate(
            prompt=rag_prompt,
            system_prompt=system_prompt,
            **kwargs,
        )