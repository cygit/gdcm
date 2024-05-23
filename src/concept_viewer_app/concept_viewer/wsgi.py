"""
WSGI config for concept_viewer project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
import sys

from django.core.wsgi import get_wsgi_application
SITE_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SITE_ROOT)
sys.path.append(BASE_DIR)
sys.path.append(SITE_ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "concept_viewer.settings")

application = get_wsgi_application()
