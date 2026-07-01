pipeline {
    agent any

    stages {
        stage('Install') {
            steps {
                sh 'mise run install'
            }
        }
        stage('Check') {
            steps {
                sh 'mise run standards:check'
            }
        }
    }
}
