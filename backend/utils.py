import re

def resolve_target(text: str, parent_username: str = None, author_username: str = None) -> str:
    """
    Dynamically identifies who a post or comment is directed toward.
    
    1. Parses @mentions from the text using rule-based regex.
    2. Excludes the author's own handle if they mention themselves.
    3. Falls back to parent_username (the post/comment being replied to) if no mentions.
    4. Falls back to '@anonymous' if no targets can be resolved.
    """
    if not text:
        return parent_username or "@anonymous"

    # Find all @handles (case insensitive comparison)
    mentions = re.findall(r"@\w+", text)
    
    # Strip whitespace/ensure standard format
    cleaned_mentions = [m.strip() for m in mentions if m.strip()]
    
    # Filter out the author themselves if they self-mention
    filtered_mentions = []
    if author_username:
        author_clean = author_username.strip().lower()
        for m in cleaned_mentions:
            if m.lower() != author_clean:
                filtered_mentions.append(m)
    else:
        filtered_mentions = cleaned_mentions

    # Return the first external @mention if found
    if filtered_mentions:
        return filtered_mentions[0]
        
    # Fallback to the parent post owner/reply-to user
    if parent_username:
        return parent_username.strip()
        
    return "@anonymous"
