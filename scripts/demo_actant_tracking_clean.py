#!/usr/bin/env python
"""
Demo script for Habitat's domain-predicate tracking with ArangoDB.
This script demonstrates how to use the repositories to track actants across domains.

NOTE: This script assumes the ArangoDB collections have already been created
using the init_arangodb.py script.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from habitat_evolution.adaptive_core.persistence.arangodb.document_repository import Document, DocumentRepository
from habitat_evolution.adaptive_core.persistence.arangodb.domain_repository import Domain, DomainRepository
from habitat_evolution.adaptive_core.persistence.arangodb.predicate_repository import Predicate, PredicateRepository
from habitat_evolution.adaptive_core.persistence.arangodb.actant_repository import Actant, ActantRepository
from habitat_evolution.adaptive_core.persistence.arangodb.pattern_evolution_tracker import PatternEvolutionTracker

# Disable any existing loggers to prevent errors
import logging
logging.disable(logging.CRITICAL)

def create_sample_data():
    """Create sample data for the demo."""
    print("Creating sample data...")
    
    # Initialize repositories
    document_repo = DocumentRepository()
    domain_repo = DomainRepository()
    predicate_repo = PredicateRepository()
    actant_repo = ActantRepository()
    
    # Create sample documents
    doc1 = Document.create(
        title="Climate Change Impact on Coral Reefs",
        content="""
        Rising ocean temperatures are causing coral reefs to bleach. 
        The coral polyps expel their symbiotic algae when stressed by high temperatures.
        Without these algae, the coral loses its main source of food and often dies.
        Scientists are studying how some coral species adapt to warmer waters.
        """,
        source="Environmental Science Journal"
    )
    doc1_id = document_repo.create(doc1)
    print(f"Created document 1: {doc1_id}")
    
    doc2 = Document.create(
        title="Marine Ecosystems and Climate Adaptation",
        content="""
        Some coral species show remarkable resilience to changing ocean conditions.
        These corals develop resistance to thermal stress through genetic adaptation.
        The symbiotic algae evolve to withstand higher temperatures, protecting their coral hosts.
        This co-evolution may help preserve reef ecosystems despite climate change.
        """,
        source="Marine Biology Research"
    )
    doc2_id = document_repo.create(doc2)
    print(f"Created document 2: {doc2_id}")
    
    # Create domains in document 1
    domain1_1 = Domain.create(
        document_id=doc1_id,
        text="Rising ocean temperatures are causing coral reefs to bleach.",
        vector=[0.1, 0.2, 0.3, 0.4],  # Simplified vector representation
        start_position=0,
        end_position=60
    )
    domain1_1_id = domain_repo.create(domain1_1)
    document_repo.link_to_domain(doc1_id, domain1_1_id)
    print(f"Created domain 1.1: {domain1_1_id}")
    
    domain1_2 = Domain.create(
        document_id=doc1_id,
        text="The coral polyps expel their symbiotic algae when stressed by high temperatures.",
        vector=[0.2, 0.3, 0.4, 0.5],
        start_position=61,
        end_position=140
    )
    domain1_2_id = domain_repo.create(domain1_2)
    document_repo.link_to_domain(doc1_id, domain1_2_id)
    print(f"Created domain 1.2: {domain1_2_id}")
    
    domain1_3 = Domain.create(
        document_id=doc1_id,
        text="Scientists are studying how some coral species adapt to warmer waters.",
        vector=[0.3, 0.4, 0.5, 0.6],
        start_position=220,
        end_position=290
    )
    domain1_3_id = domain_repo.create(domain1_3)
    document_repo.link_to_domain(doc1_id, domain1_3_id)
    print(f"Created domain 1.3: {domain1_3_id}")
    
    # Create domains in document 2
    domain2_1 = Domain.create(
        document_id=doc2_id,
        text="Some coral species show remarkable resilience to changing ocean conditions.",
        vector=[0.3, 0.4, 0.5, 0.6],  # Similar to domain1_3
        start_position=0,
        end_position=70
    )
    domain2_1_id = domain_repo.create(domain2_1)
    document_repo.link_to_domain(doc2_id, domain2_1_id)
    print(f"Created domain 2.1: {domain2_1_id}")
    
    domain2_2 = Domain.create(
        document_id=doc2_id,
        text="The symbiotic algae evolve to withstand higher temperatures, protecting their coral hosts.",
        vector=[0.4, 0.5, 0.6, 0.7],
        start_position=140,
        end_position=220
    )
    domain2_2_id = domain_repo.create(domain2_2)
    document_repo.link_to_domain(doc2_id, domain2_2_id)
    print(f"Created domain 2.2: {domain2_2_id}")
    
    # Create actants
    coral_actant = actant_repo.find_or_create("coral", ["coral reefs", "coral species", "coral polyps", "coral hosts"])
    print(f"Created/found coral actant: {coral_actant.id}")
    
    algae_actant = actant_repo.find_or_create("symbiotic algae", ["algae"])
    print(f"Created/found algae actant: {algae_actant.id}")
    
    temperature_actant = actant_repo.find_or_create("temperature", ["ocean temperatures", "high temperatures", "warmer waters", "higher temperatures"])
    print(f"Created/found temperature actant: {temperature_actant.id}")
    
    scientists_actant = actant_repo.find_or_create("scientists")
    print(f"Created/found scientists actant: {scientists_actant.id}")
    
    # Create predicates in domain 1.1
    pred1_1 = Predicate.create(
        domain_id=domain1_1_id,
        subject="ocean temperatures",
        verb="causing",
        object="coral reefs",
        text="Rising ocean temperatures are causing coral reefs to bleach",
        position=0
    )
    pred1_1_id = predicate_repo.create(pred1_1)
    predicate_repo.link_to_domain(pred1_1_id, domain1_1_id)
    print(f"Created predicate 1.1: {pred1_1_id}")
    
    # Link actants to predicate 1.1
    actant_repo.link_to_predicate(temperature_actant.id, pred1_1_id, "subject")
    actant_repo.link_to_predicate(coral_actant.id, pred1_1_id, "object")
    
    # Create predicates in domain 1.2
    pred1_2 = Predicate.create(
        domain_id=domain1_2_id,
        subject="coral polyps",
        verb="expel",
        object="symbiotic algae",
        text="The coral polyps expel their symbiotic algae when stressed",
        position=0
    )
    pred1_2_id = predicate_repo.create(pred1_2)
    predicate_repo.link_to_domain(pred1_2_id, domain1_2_id)
    print(f"Created predicate 1.2: {pred1_2_id}")
    
    # Link actants to predicate 1.2
    actant_repo.link_to_predicate(coral_actant.id, pred1_2_id, "subject")
    actant_repo.link_to_predicate(algae_actant.id, pred1_2_id, "object")
    
    # Create predicates in domain 1.3
    pred1_3 = Predicate.create(
        domain_id=domain1_3_id,
        subject="coral species",
        verb="adapt",
        object="warmer waters",
        text="Scientists are studying how some coral species adapt to warmer waters",
        position=0
    )
    pred1_3_id = predicate_repo.create(pred1_3)
    predicate_repo.link_to_domain(pred1_3_id, domain1_3_id)
    print(f"Created predicate 1.3: {pred1_3_id}")
    
    # Link actants to predicate 1.3
    actant_repo.link_to_predicate(coral_actant.id, pred1_3_id, "subject")
    actant_repo.link_to_predicate(temperature_actant.id, pred1_3_id, "object")
    
    # Create predicates in domain 2.1
    pred2_1 = Predicate.create(
        domain_id=domain2_1_id,
        subject="coral species",
        verb="show",
        object="resilience",
        text="Some coral species show remarkable resilience to changing ocean conditions",
        position=0
    )
    pred2_1_id = predicate_repo.create(pred2_1)
    predicate_repo.link_to_domain(pred2_1_id, domain2_1_id)
    print(f"Created predicate 2.1: {pred2_1_id}")
    
    # Link actants to predicate 2.1
    actant_repo.link_to_predicate(coral_actant.id, pred2_1_id, "subject")
    
    # Create predicates in domain 2.2
    pred2_2 = Predicate.create(
        domain_id=domain2_2_id,
        subject="symbiotic algae",
        verb="evolve",
        object="higher temperatures",
        text="The symbiotic algae evolve to withstand higher temperatures",
        position=0
    )
    pred2_2_id = predicate_repo.create(pred2_2)
    predicate_repo.link_to_domain(pred2_2_id, domain2_2_id)
    print(f"Created predicate 2.2: {pred2_2_id}")
    
    # Link actants to predicate 2.2
    actant_repo.link_to_predicate(algae_actant.id, pred2_2_id, "subject")
    actant_repo.link_to_predicate(temperature_actant.id, pred2_2_id, "object")
    
    pred2_3 = Predicate.create(
        domain_id=domain2_2_id,
        subject="symbiotic algae",
        verb="protecting",
        object="coral hosts",
        text="The symbiotic algae protecting their coral hosts",
        position=50
    )
    pred2_3_id = predicate_repo.create(pred2_3)
    predicate_repo.link_to_domain(pred2_3_id, domain2_2_id)
    print(f"Created predicate 2.3: {pred2_3_id}")
    
    # Link actants to predicate 2.3
    actant_repo.link_to_predicate(algae_actant.id, pred2_3_id, "subject")
    actant_repo.link_to_predicate(coral_actant.id, pred2_3_id, "object")
    
    print("Sample data creation complete!")
    
    return {
        "documents": [doc1_id, doc2_id],
        "domains": [domain1_1_id, domain1_2_id, domain1_3_id, domain2_1_id, domain2_2_id],
        "predicates": [pred1_1_id, pred1_2_id, pred1_3_id, pred2_1_id, pred2_2_id, pred2_3_id],
        "actants": {
            "coral": coral_actant.id,
            "algae": algae_actant.id,
            "temperature": temperature_actant.id,
            "scientists": scientists_actant.id
        }
    }

def track_actant_evolution(actant_name: str):
    """Track and display how an actant evolves across domains."""
    print(f"Tracking evolution of actant: {actant_name}")
    
    tracker = PatternEvolutionTracker()
    
    # Get the actant's journey
    journey = tracker.track_actant_journey(actant_name)
    
    print(f"\n===== ACTANT JOURNEY: {actant_name.upper()} =====")
    print(f"Found {len(journey)} appearances across domains\n")
    
    for i, step in enumerate(journey):
        domain = step.get("domain", {})
        predicate = step.get("predicate", {})
        document = step.get("document", {})
        role = step.get("role", "")
        
        print(f"Step {i+1}:")
        print(f"  Document: {document.get('title', 'Unknown')}")
        print(f"  Domain: {domain.get('text', 'Unknown')[:50]}...")
        print(f"  Role: {role}")
        
        if role == "subject":
            print(f"  Predicate: {predicate.get('subject', '')} {predicate.get('verb', '')} {predicate.get('object', '')}")
        else:
            print(f"  Predicate: {predicate.get('subject', '')} {predicate.get('verb', '')} {predicate.get('object', '')}")
        
        print("")
    
    # Get predicate transformations
    transformations = tracker.detect_predicate_transformations(actant_name)
    
    print(f"\n===== PREDICATE TRANSFORMATIONS FOR {actant_name.upper()} =====")
    if not transformations:
        print("No transformations found")
        return
        
    print(f"Found {len(transformations)} transformations\n")
    
    for i, transform in enumerate(transformations):
        from_pred = transform.get("from", {}).get("predicate", {})
        to_pred = transform.get("to", {}).get("predicate", {})
        
        print(f"Transformation {i+1}:")
        print(f"  From: {from_pred.get('subject', '')} {from_pred.get('verb', '')} {from_pred.get('object', '')}")
        print(f"  To:   {to_pred.get('subject', '')} {to_pred.get('verb', '')} {to_pred.get('object', '')}")
        
        if transform.get("verb_change"):
            print(f"  Verb changed: {from_pred.get('verb', '')} → {to_pred.get('verb', '')}")
            
        if transform.get("role_change"):
            from_role = transform.get("from", {}).get("role", "")
            to_role = transform.get("to", {}).get("role", "")
            print(f"  Role changed: {from_role} → {to_role}")
            
        print("")

def find_resonating_domains():
    """Find and display domains that resonate with each other."""
    print("Finding resonating domains...")
    
    tracker = PatternEvolutionTracker()
    resonances = tracker.find_resonating_domains(threshold=0.3)
    
    print(f"\n===== RESONATING DOMAINS =====")
    print(f"Found {len(resonances)} resonating domain pairs\n")
    
    for i, resonance in enumerate(resonances):
        domain1 = resonance.get("domain1", {})
        domain2 = resonance.get("domain2", {})
        similarity = resonance.get("similarity", 0)
        shared_actants = resonance.get("shared_actants", [])
        
        print(f"Resonance {i+1} (similarity: {similarity:.2f}):")
        print(f"  Domain 1: {domain1.get('text', 'Unknown')[:50]}...")
        print(f"  Domain 2: {domain2.get('text', 'Unknown')[:50]}...")
        print(f"  Shared actants: {', '.join(shared_actants)}")
        print("")
        
        # Record the resonance in the database
        tracker.record_domain_resonance(
            domain1.get("_key"), 
            domain2.get("_key"),
            similarity,
            shared_actants
        )

def main():
    """Main function to run the demo."""
    print("\n===== HABITAT DOMAIN-PREDICATE TRACKING DEMO =====")
    print("This demo shows how Habitat tracks actants across domains and detects evolutionary patterns.\n")
    
    try:
        # Create sample data
        data = create_sample_data()
        
        # Track coral actant evolution
        track_actant_evolution("coral")
        
        # Track algae actant evolution
        track_actant_evolution("symbiotic algae")
        
        # Find resonating domains
        find_resonating_domains()
        
        print("\n===== DEMO COMPLETE =====")
        print("The ArangoDB database now contains a sample of domain-predicate tracking data.")
        print("You can explore this data further using the ArangoDB web interface at http://localhost:8529")
        print("Username: root, Password: habitat, Database: habitat")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPlease make sure ArangoDB is running and the collections have been created.")
        print("You can create the collections using the init_arangodb.py script.")

if __name__ == "__main__":
    main()
