# EmailBot

Local scraper for job-application email. Sign in yourself in the browser window, then the script reads Gmail and saves a spreadsheet.

```powershell
pip install -r requirements.txt
python scrape_job_applications.py
```

Results land in `output/job_applications.csv` with date, company, role, and status. The signed-in browser profile, scraped mail, labeled dataset, and trained model stay on your machine.

The first pass is a keyword script. An email is kept when it says thank or thanks, and also application or applying. That still lets through mail that uses those words and is not about your application. Companies format follow-ups differently, so the keywords cannot cover every case. Training a model on your own labeled export is how you filter those.

## Train your own classifier

The scraper starts with keyword rules. After you label your own export, a TF-IDF linear classifier (logistic regression) can decide which emails are real applications and which status they have.

```powershell
python prepare_dataset.py
```

That writes two local files:

- `data/labeled.jsonl` — emails the keyword rules already marked as applied, rejected, interview, or online assessment
- `data/review.csv` — unknown emails for you to check

In `review.csv`, set `label` to one of:

- `applied`
- `rejected`
- `interview`
- `online assessment`
- `no`

`no` means the message is not an application. `suggestion` is only a hint. Leave a row blank if you are still unsure, and it stays out of training. Run `prepare_dataset.py` again after editing the labels.

```powershell
python train_model.py
```

Training prints how the keyword rules score against your labels, then the same check for the model on held-out rows. The model is saved to `models/email_classifier.joblib`.

The next scrape uses that model to drop mail that is not an application. Status stays with the keyword rules unless the status model scores higher on held-out mail. Delete the model file to go back to keywords only.

Training also writes three PNGs to `performance/`:

- `accuracy_comparison.png` — keyword rules next to the TF-IDF model
- `application_confusion.png` — held-out application vs not
- `status_confusion.png` — held-out status, when there are enough labeled emails

Those pictures are scores only, with no message text, so they can be committed for anyone reading the repo. They describe the inbox that was labeled for that run. Another person gets different charts from their own mail, and they never receive the model file.
