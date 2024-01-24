# Install required packages
# pip install transformers datasets torch

from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import torch

# Step 1: Prepare the Dataset
data = [
    {"text": "Tester must submit bug report", "keywords": "Tester, submit, bug, report"},
    {"text": "Developer must assign bug", "keywords": "Developer, assign, bug"},
    {"text": "Tester must verify fix", "keywords": "Tester, verify, fix"},
    {"text": "Project manager must generate reports", "keywords": "Project manager, generate, reports"},
    {"text": "Customer must submit ticket", "keywords": "Customer, submit, ticket"},
    {"text": "Support agent must resolve issue", "keywords": "Support agent, resolve, issue"},
    {"text": "Manager must escalate ticket", "keywords": "Manager, escalate, ticket"},
    {"text": "Manager must monitor performance", "keywords": "Manager, monitor, performance"},
    {"text": "Organizer must create event proposal", "keywords": "Organizer, create, event, proposal"},
    {"text": "Organizer must hire vendors", "keywords": "Organizer, hire, vendors"},
    {"text": "Client must approve proposal", "keywords": "Client, approve, proposal"},
    {"text": "Vendor must coordinate logistics", "keywords": "Vendor, coordinate, logistics"},
    {"text": "User must set fitness goals", "keywords": "User, set, fitness, goals"},
    {"text": "User must track workouts", "keywords": "User, track, workouts"},
    {"text": "App must generate reports", "keywords": "App, generate, reports"},
    {"text": "Trainer must provide guidance", "keywords": "Trainer, provide, guidance"},
    {"text": "Patient must record vital signs", "keywords": "Patient, record, vital, signs"},
    {"text": "Provider must record vital signs", "keywords": "Provider, record, vital, signs"},
    {"text": "Patient must schedule appointments", "keywords": "Patient, schedule, appointments"},
    {"text": "Provider must schedule appointments", "keywords": "Provider, schedule, appointments"},
    {"text": "Patient must view health history", "keywords": "Patient, view, health, history"},
    {"text": "Provider must view health history", "keywords": "Provider, view, health, history"},
    {"text": "System must alert emergency services", "keywords": "System, alert, emergency, services"},
    {"text": "Guest must book room", "keywords": "Guest, book, room"},
    {"text": "Guest must check-in with receptionist", "keywords": "Guest, check-in, receptionist"},
    {"text": "Receptionist must check-in guest", "keywords": "Receptionist, check-in, guest"},
    {"text": "Guest must check-out with receptionist", "keywords": "Guest, check-out, receptionist"},
    {"text": "Receptionist must check-out guest", "keywords": "Receptionist, check-out, guest"},
    {"text": "Manager must manage reservations", "keywords": "Manager, manage, reservations"},
    {"text": "Warehouse manager must track inventory", "keywords": "Warehouse manager, track, inventory"},
    {"text": "Supplier must order stock", "keywords": "Supplier, order, stock"},
    {"text": "Sales team must update prices", "keywords": "Sales team, update, prices"},
    {"text": "Warehouse manager must generate reports", "keywords": "Warehouse manager, generate, reports"},
    {"text": "Student must enroll in course", "keywords": "Student, enroll, course"},
    {"text": "Student must submit assignments", "keywords": "Student, submit, assignments"},
    {"text": "Instructor must grade assignments", "keywords": "Instructor, grade, assignments"},
    {"text": "Administrator must manage courses", "keywords": "Administrator, manage, courses"},
    {"text": "Librarian must add book to catalog", "keywords": "Librarian, add, book, catalog"},
    {"text": "Member must check out book", "keywords": "Member, check out, book"},
    {"text": "Member must return book", "keywords": "Member, return, book"},
    {"text": "Member must renew membership", "keywords": "Member, renew, membership"},
    {"text": "Manager must create project", "keywords": "Manager, create, project"},
    {"text": "Manager must assign tasks", "keywords": "Manager, assign, tasks"},
    {"text": "Stakeholder must generate reports", "keywords": "Stakeholder, generate, reports"},
    {"text": "Team member must collaborate", "keywords": "Team member, collaborate"},
    {"text": "Team member must review progress", "keywords": "Team member, review, progress"},
    {"text": "Customer must place order", "keywords": "Customer, place, order"},
    {"text": "Chef must prepare food", "keywords": "Chef, prepare, food"},
    {"text": "Waiter must serve order", "keywords": "Waiter, serve, order"},
    {"text": "Chef must manage menu", "keywords": "Chef, manage, menu"},
    {"text": "User must create post", "keywords": "User, create, post"},
    {"text": "User must like/dislike post", "keywords": "User, like, dislike, post"},
    {"text": "User must report content", "keywords": "User, report, content"},
    {"text": "Moderator must moderate content", "keywords": "Moderator, moderate, content"},
    {"text": "Admin must administrate users", "keywords": "Admin, administrate, users"},
    {"text": "Team leader must create task", "keywords": "Team leader, create, task"},
    {"text": "Team leader must assign task", "keywords": "Team leader, assign, task"},
    {"text": "Team member must complete task", "keywords": "Team member, complete, task"},
    {"text": "Project manager must generate reports", "keywords": "Project manager, generate, reports"},
    {"text": "Traveler must search flights", "keywords": "Traveler, search, flights"},
    {"text": "Traveler must book ticket with agent", "keywords": "Traveler, book, ticket, agent"},
    {"text": "Agent must book ticket for traveler", "keywords": "Agent, book, ticket, traveler"},
    {"text": "Traveler must cancel reservation with agent", "keywords": "Traveler, cancel, reservation, agent"},
    {"text": "Agent must cancel reservation for traveler", "keywords": "Agent, cancel, reservation, traveler"},
    {"text": "Traveler must manage preferences", "keywords": "Traveler, manage, preferences"},
    {"text": "System must generate itinerary", "keywords": "System, generate, itinerary"},
    {"text": "Voter must register to vote", "keywords": "Voter, register, vote"},
    {"text": "Election commission must conduct election", "keywords": "Election commission, conduct, election"},
    {"text": "Candidate must count votes", "keywords": "Candidate, count, votes"},
    {"text": "Election commission must declare results", "keywords": "Election commission, declare, results"},
    {"text": "Guest must search catalog", "keywords": "Guest, search, catalog"},
    {"text": "Member must check out book", "keywords": "Member, check out, book"},
    {"text": "Member must return book", "keywords": "Member, return, book"},
    {"text": "Member must renew book", "keywords": "Member, renew, book"},
    {"text": "Librarian must manage fines", "keywords": "Librarian, manage, fines"},
    {"text": "Librarian must add new book", "keywords": "Librarian, add, new, book"},
    {"text": "Guest must browse products", "keywords": "Guest, browse, products"},
    {"text": "User must add to cart", "keywords": "User, add, cart"},
    {"text": "User must remove from cart", "keywords": "User, remove, cart"},
    {"text": "User must checkout", "keywords": "User, checkout"},
    {"text": "Admin must manage inventory", "keywords": "Admin, manage, inventory"},
    {"text": "Gateway must process payment", "keywords": "Gateway, process, payment"},
    {"text": "User must handle user logins securely", "keywords": "User, handle, logins, securely"},
    {"text": "User must allow file uploads", "keywords": "User, allow, file, uploads"}
]


transformed_data = {
    "text": [item["text"] for item in data],
    "keywords": [item["keywords"] for item in data]
}

dataset = Dataset.from_dict(transformed_data)

# Tokenize the Dataset
tokenizer = T5Tokenizer.from_pretrained('t5-small')

def tokenize_function(examples):
    model_inputs = tokenizer(examples["keywords"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512).input_ids
    model_inputs["labels"] = labels
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load a Pre-trained Model
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set the device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)

# Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,  # Increased number of epochs
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # Disable wandb integration
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the Model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")