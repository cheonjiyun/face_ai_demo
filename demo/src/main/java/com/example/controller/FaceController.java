@PostMapping("/api/face-analyze")
public ResponseEntity<?> analyzeFace(@RequestParam MultipartFile image) {
    HttpHeaders headers = new HttpHeaders();
    headers.setContentType(MediaType.MULTIPART_FORM_DATA);
    
    MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
    body.add("file", new ByteArrayResource(image.getBytes()) {
        @Override
        public String getFilename() {
            return image.getOriginalFilename();
        }
    });

    HttpEntity<MultiValueMap<String, Object>> request = new HttpEntity<>(body, headers);
    RestTemplate restTemplate = new RestTemplate();
    ResponseEntity<String> response = restTemplate.postForEntity(
        "http://localhost:5000/analyze", request, String.class
    );

    return ResponseEntity.ok(response.getBody());
}
