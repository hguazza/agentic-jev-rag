#!/usr/bin/env bash
# Build the api and voice images for arm64 and push them to ECR, tagged with the current commit.
# Tags are immutable, so an image already pushed for this commit is skipped. Prints the tag last.
set -euo pipefail
cd "$(dirname "$0")/../.."

git diff --quiet HEAD -- || { echo "Commit your changes first: images are tagged with the commit." >&2; exit 1; }
tag=$(git rev-parse --short=12 HEAD)
region=${AWS_REGION:-us-east-1}
registry="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${region}.amazonaws.com"
aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$registry" >/dev/null

for target in api voice; do
  if aws ecr describe-images --region "$region" --repository-name "swiftai/$target" --image-ids imageTag="$tag" >/dev/null 2>&1; then
    echo "swiftai/$target:$tag already pushed" >&2
    continue
  fi
  docker build --platform linux/arm64 --target "$target" -t "$registry/swiftai/$target:$tag" . >&2
  docker push "$registry/swiftai/$target:$tag" >&2
done
echo "$tag"
