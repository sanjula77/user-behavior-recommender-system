#!/usr/bin/env python3
"""
Generate sample web behavior events for testing ML models.
Creates realistic user sessions with page views, clicks, and metadata.

Usage:
    python ml/generate_sample_data.py --users 50 --save-mongo --save-csv
    python ml/generate_sample_data.py --users 10 --days 7  # Generate 7 days of data
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
import os
import sys
import csv
import json

# Try to import pandas (optional - only needed for advanced CSV features)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ml.pipeline.db_client import get_events_collection
    MONGO_AVAILABLE = True
except Exception as e:
    MONGO_AVAILABLE = False
    print(f"Warning: MongoDB not available: {e}. Will only save to CSV.")


# Sample page URLs (realistic web application pages)
PAGE_URLS = [
    "https://example.com/",
    "https://example.com/products",
    "https://example.com/products/laptop",
    "https://example.com/products/phone",
    "https://example.com/products/tablet",
    "https://example.com/products/headphones",
    "https://example.com/about",
    "https://example.com/contact",
    "https://example.com/cart",
    "https://example.com/checkout",
    "https://example.com/login",
    "https://example.com/signup",
    "https://example.com/account",
    "https://example.com/orders",
    "https://example.com/settings",
    "https://example.com/help",
    "https://example.com/blog",
    "https://example.com/blog/article-1",
    "https://example.com/blog/article-2",
    "https://example.com/search?q=laptop",
    "https://example.com/search?q=phone",
    "https://example.com/category/electronics",
    "https://example.com/category/accessories",
    "https://example.com/product/laptop/dell-xps",
    "https://example.com/product/phone/iphone-15",
]

# Common click elements
CLICK_ELEMENTS = [
    {"tag": "BUTTON", "id": "add-to-cart", "classes": "btn btn-primary"},
    {"tag": "A", "id": "product-link", "classes": "product-card"},
    {"tag": "BUTTON", "id": "buy-now", "classes": "btn btn-success"},
    {"tag": "A", "id": "next-page", "classes": "pagination-link"},
    {"tag": "BUTTON", "id": "subscribe", "classes": "btn btn-secondary"},
    {"tag": "A", "id": "read-more", "classes": "article-link"},
    {"tag": "BUTTON", "id": "apply-filter", "classes": "filter-btn"},
    {"tag": "A", "id": "related-product", "classes": "related-item"},
]


def generate_user_id() -> str:
    """Generate a realistic user ID."""
    return f"user-{random.randint(1000, 9999)}"


def generate_session_id(user_id: str) -> str:
    """Generate a session ID for a user."""
    return f"{user_id}_sess_{random.randint(100, 999)}"


def generate_timestamp(base_time: datetime, offset_seconds: int = 0) -> str:
    """Generate ISO format timestamp."""
    dt = base_time + timedelta(seconds=offset_seconds)
    return dt.isoformat()


def generate_page_view_event(
    user_id: str,
    session_id: str,
    page_url: str,
    timestamp: str
) -> Dict[str, Any]:
    """Generate a page_view event."""
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "event_type": "page_view",
        "page_url": page_url,
        "timestamp": timestamp,
        "metadata": {}
    }


def generate_click_event(
    user_id: str,
    session_id: str,
    page_url: str,
    timestamp: str,
    click_metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Generate a click event."""
    if click_metadata is None:
        click_metadata = random.choice(CLICK_ELEMENTS)
    
    return {
        "event_id": str(uuid.uuid4()),
        "user_id": user_id,
        "session_id": session_id,
        "event_type": "click",
        "page_url": page_url,
        "timestamp": timestamp,
        "metadata": click_metadata
    }


def generate_user_session(
    user_id: str,
    session_id: str,
    start_time: datetime,
    duration_minutes: int = None
) -> List[Dict[str, Any]]:
    """
    Generate a complete user session with realistic behavior.
    
    Args:
        user_id: User identifier
        session_id: Session identifier
        start_time: When the session starts
        duration_minutes: How long the session lasts (random if None)
    """
    events = []
    
    if duration_minutes is None:
        duration_minutes = random.randint(2, 30)  # 2-30 minute sessions
    
    # Session starts with a page view
    current_url = random.choice(PAGE_URLS)
    events.append(generate_page_view_event(
        user_id, session_id, current_url,
        generate_timestamp(start_time, 0)
    ))
    
    # Generate browsing behavior
    time_offset = random.randint(5, 30)  # Time on first page: 5-30 seconds
    pages_visited = random.randint(3, 12)  # 3-12 pages per session
    
    for i in range(pages_visited - 1):
        # Decide action: click or navigate directly
        if random.random() < 0.7:  # 70% chance of click before navigation
            # Generate 1-3 clicks on current page
            num_clicks = random.randint(1, 3)
            for click_idx in range(num_clicks):
                time_offset += random.randint(2, 10)  # Time between clicks
                events.append(generate_click_event(
                    user_id, session_id, current_url,
                    generate_timestamp(start_time, time_offset)
                ))
        
        # Navigate to next page
        time_offset += random.randint(10, 60)  # Time spent on page before navigation
        
        # Choose next page (sometimes related, sometimes random)
        if random.random() < 0.6:  # 60% chance to go to related page
            # Related pages (simple heuristic: same domain, related paths)
            related_urls = [url for url in PAGE_URLS if 
                          url.split('/')[-1] in current_url or 
                          current_url.split('/')[-1] in url or
                          random.random() < 0.3]
            if related_urls:
                current_url = random.choice(related_urls)
            else:
                current_url = random.choice(PAGE_URLS)
        else:
            current_url = random.choice(PAGE_URLS)
        
        # Page view event
        if time_offset < duration_minutes * 60:  # Don't exceed session duration
            events.append(generate_page_view_event(
                user_id, session_id, current_url,
                generate_timestamp(start_time, time_offset)
            ))
    
    # Add some final clicks on last page
    if random.random() < 0.5 and len(events) > 0:
        final_clicks = random.randint(1, 2)
        for _ in range(final_clicks):
            time_offset += random.randint(5, 20)
            if time_offset < duration_minutes * 60:
                events.append(generate_click_event(
                    user_id, session_id, current_url,
                    generate_timestamp(start_time, time_offset)
                ))
    
    return events


def generate_sample_data(
    n_users: int = 50,
    days_back: int = 30,
    sessions_per_user: int = None,
    save_mongo: bool = False,
    save_csv: bool = True
) -> List[Dict[str, Any]]:
    """
    Generate sample web behavior events.
    
    Args:
        n_users: Number of unique users
        days_back: How many days back to generate data
        sessions_per_user: Average sessions per user (random if None)
        save_mongo: Save to MongoDB if available
        save_csv: Save to CSV file
    """
    print(f"Generating sample data for {n_users} users over {days_back} days...")
    
    all_events = []
    users = [generate_user_id() for _ in range(n_users)]
    
    if sessions_per_user is None:
        sessions_per_user = random.randint(2, 10)  # 2-10 sessions per user on average
    
    # Generate sessions for each user over the time period
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    for user_id in users:
        # Each user has variable number of sessions
        user_sessions = random.randint(max(1, sessions_per_user - 3), sessions_per_user + 3)
        
        for session_num in range(user_sessions):
            # Random session start time within the period
            session_start_offset = random.randint(0, days_back * 24 * 60 * 60)
            session_start = start_time + timedelta(seconds=session_start_offset)
            
            session_id = generate_session_id(user_id)
            session_events = generate_user_session(user_id, session_id, session_start)
            all_events.extend(session_events)
    
    print(f"Generated {len(all_events)} events across {len(users)} users")
    
    # Save to MongoDB if requested and available
    if save_mongo and MONGO_AVAILABLE:
        try:
            collection = get_events_collection()
            # Remove _id if present to avoid conflicts
            events_to_insert = []
            for event in all_events:
                event_copy = event.copy()
                if '_id' in event_copy:
                    del event_copy['_id']
                events_to_insert.append(event_copy)
            
            result = collection.insert_many(events_to_insert)
            print(f"✓ Saved {len(result.inserted_ids)} events to MongoDB")
        except Exception as e:
            print(f"✗ Failed to save to MongoDB: {e}")
    
    # Save to CSV
    if save_csv:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "events_export.csv")
        
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(all_events)
            # Ensure timestamp is string for CSV
            df['timestamp'] = df['timestamp'].astype(str)
            df.to_csv(csv_path, index=False)
            print(f"✓ Saved {len(df)} events to {csv_path}")
        else:
            # Use standard library csv module
            if all_events:
                fieldnames = all_events[0].keys()
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for event in all_events:
                        # Convert metadata dict to JSON string for CSV
                        event_copy = event.copy()
                        if isinstance(event_copy.get('metadata'), dict):
                            event_copy['metadata'] = json.dumps(event_copy['metadata'])
                        writer.writerow(event_copy)
                print(f"✓ Saved {len(all_events)} events to {csv_path}")
    
    return all_events


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample web behavior events for ML model training"
    )
    parser.add_argument(
        "--users", "-u",
        type=int,
        default=50,
        help="Number of unique users (default: 50)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=30,
        help="Number of days back to generate data (default: 30)"
    )
    parser.add_argument(
        "--sessions-per-user", "-s",
        type=int,
        default=None,
        help="Average sessions per user (random if not specified)"
    )
    parser.add_argument(
        "--save-mongo",
        action="store_true",
        help="Save events to MongoDB (requires DB connection)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        default=True,
        help="Save events to CSV file (default: True)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Don't save to CSV"
    )
    
    args = parser.parse_args()
    
    if args.no_csv:
        args.save_csv = False
    
    generate_sample_data(
        n_users=args.users,
        days_back=args.days,
        sessions_per_user=args.sessions_per_user,
        save_mongo=args.save_mongo,
        save_csv=args.save_csv
    )


if __name__ == "__main__":
    main()

