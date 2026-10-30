# EmailBot
lightweight local AI powered Email bot - not yet AI powered

Currently using just a Python script to parse emails to identify job application related emails.
Easiest way to identify those emails is if they contain the words (applying, application, thank you) all in the same email. A further check is that email also contains the words (talent acquisition and career)
This still does not handle cases where a non job application related email contains the 3 key words. This is the difference between regular workflow and an AI workflow. Different companies have different email follow-ups and formatting, so simple python script cannot handle all edge cases. As a last resort use a model to either filter out non job application emails or to identify role/position titles.
