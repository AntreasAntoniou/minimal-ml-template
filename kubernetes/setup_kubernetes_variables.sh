#!/bin/bash
gcloud config set project $CLUSTER_PROJECT
gcloud container clusters get-credentials $CLUSTER_NAME \
    --zone $CLUSTER_ZONE --project $CLUSTER_PROJECT
