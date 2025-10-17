from __future__ import annotations
import socket
import smtplib
import ssl
import ast
import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from dotenv import dotenv_values

def send_notification(
        subject, 
        body,
        sender_email, 
        receiver_email, 
        smtp_server, 
        smtp_port, 
        password, 
        results_fname=None, 
        results_fpath=None):

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))  

    if results_fname and results_fpath:
        try:
            with open(results_fpath, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{results_fname}"')
            msg.attach(part)
        except FileNotFoundError:
            return f"Error: File '{results_fname}' not found at path '{results_fpath}'."

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, password)
            server.sendmail(sender_email, [receiver_email], msg.as_string())
        return "Email sent successfully"
    except smtplib.SMTPException as e:
        return f"SMTP error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"


class Notify():
    def __init__(
            self, 
            job_name: str,
            sender_email: str, 
            receiver_emails: Sequence[str], 
            smtp_server: str, 
            smtp_port: str, 
            password: str,
            logger: Optional[logging.Logger] = None
            ):
        self.job_name = job_name
        self.sender_email = sender_email
        self.receiver_emails = receiver_emails
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.password = password
        self.logger = logger
        self._attachment: Optional[Tuple[str, str]] = None

    def attach(self, filename: str, filepath: str):
        self._attachment = (filename, filepath)

    def __enter__(self):
        if self.logger: 
            self.logger.info(f"Training '{self.job_name}' started on {socket.gethostname()}")
        return self
    
    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is None:
            subject = f"[OK] Training Finished: {self.job_name}"
            body = (
                f"Hello.\n{socket.gethostname()} has finished the training job '{self.job_name}'.\n"
                f"Thanks\ninfo.training.johnston"
            )
            fname, fpath = (self._attachment or (None, None))
        else:
            subject = f"[FAIL] Training incomplete: {self.job_name}"
            body = (
                f"Hello.\n{socket.gethostname()} couldn't finish the training job '{self.job_name}'.\n"
                f"Error: {exec_value!r}\n\n"
                f"Thanks\ninfo.training.johnston"
            )
            fname, fpath = (None, None)
        
        for receiver in self.receiver_emails:
            try: 
                response = send_notification(
                    subject=subject, 
                    body=body,
                    sender_email=self.sender_email, 
                    receiver_email=receiver, 
                    smtp_server=self.smtp_server, 
                    smtp_port=self.smtp_port, 
                    password=self.password, 
                    results_fname=fname, 
                    results_fpath=fpath
                )
                if self.logger:
                    self.logger.info(f"[notify] {response} -> {receiver}")
            except Exception as e:
                if self.logger:
                    self.logger.exception(f"[notify] Failed to notify {receiver}: {e}")
        return False


if __name__=="__main__":
    env_vars = dotenv_values(dotenv_path="./.env")
    interrupted = False
    Path(env_vars["output_folder_path"]).mkdir(parents=True, exist_ok=True)
    model_name, model_config = str(env_vars["model_name"]), str(env_vars["model_config"])
    job_name = f"{model_name}_{model_config}.{env_vars['variant']}"

    params = {
        "job_name": job_name,
        "sender_email": env_vars.get("sender_email"),
        "receiver_emails": ast.literal_eval(env_vars.get("receiver_emails", "[]")),
        "smtp_server": env_vars.get("smtp_server"),
        "smtp_port": int(env_vars.get("smtp_port", "587")),
        "password": env_vars.get("password"),
        "logger": None,
        }

    
    with Notify(**params) as notifier:
        try:
            print("OK")
        
        except KeyboardInterrupt:
            interrupted = True
            raise


        