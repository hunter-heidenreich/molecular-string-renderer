---
applyTo: "**"
---

# GitHub Copilot Instructions

NO EMOJIS

NO EMOJIS

NO EMOJIS

NO EMOJIS

This is a Python repo for rendering molecular structures from various string formats (e.g., SMILES, InChI) to images using RDKit and Pillow.

- use logging instead of print statements
- write type hints for all functions and methods; use the most modern PEP practices when performing your type hinting (no `List`, `Dict`, etc. from `typing` - use `list`, `dict`, etc. instead)
- no relative imports; always use absolute imports
- write docstrings for all functions and methods using the Google style
- use f-strings for string formatting, but only when you need to embed expressions; otherwise, use regular strings
- use `pathlib.Path` for all file path manipulations
- when opening files, always use a context manager (the `with` statement)
- use comments only for complex or non-obvious code; avoid redundant comments that state the obvious

## SOLID principles to follow:

- Single Responsibility Principle: Each class and function should have one responsibility or reason to change.
- Open/Closed Principle: Classes should be open for extension but closed for modification. Use inheritance and interfaces to achieve this.
- Liskov Substitution Principle: Subtypes must be substitutable for their base types without altering the correctness of the program.
- Interface Segregation Principle: Prefer many specific interfaces over a single general-purpose interface. Clients should not be forced to depend on interfaces they do not use.
- Dependency Inversion Principle: Depend on abstractions, not on concrete implementations.

## DRY principle to follow:

- Don't Repeat Yourself: Avoid code duplication by abstracting common functionality into reusable functions or classes.
- Use loops, functions, and classes to encapsulate repeated logic.
- Leverage inheritance and composition to share behavior among classes.
- Use configuration files or constants to manage repeated values or settings.
- Refactor code regularly to eliminate redundancy and improve maintainability.

## YAGNI principle to follow:

- You Aren't Gonna Need It: Avoid adding functionality until it is necessary. Focus on the current requirements and avoid speculative features.
- Keep the codebase simple and avoid over-engineering.
- Prioritize features based on actual user needs and feedback rather than assumptions about future requirements.
- Regularly review and refactor the code to remove unused or unnecessary components.
- Write tests to validate the necessity of features and ensure that only required functionality is implemented.

## General best practices:

- If something can be done in a simpler way, do it that way.
- If something can be refactored to be cleaner or clearer, do it.
- If something can be optimized for performance without sacrificing readability, do it.

Be clear, concise, and maintainable in your code suggestions.

When code violates any of these principles, please suggest a refactor that adheres to them.

## Testing best practices:

- Write unit tests for all functions and methods, covering both typical and edge cases.
- Do not test implementation details; focus on testing the public interface and behavior.
- Use descriptive names for test functions and classes to clearly indicate what is being tested.
- Seek to achieve high code coverage, but prioritize meaningful tests over coverage percentage.
- Attempt to follow the Arrange-Act-Assert (AAA) pattern in your tests.
- Do not test trivial aspects of code, do not test built-in language features, and do not test third-party libraries.
- Use the design patterns of the codebase to your advantage when writing tests.
